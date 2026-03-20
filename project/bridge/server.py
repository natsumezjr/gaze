# WebSocket桥接服务器 - 将进程内事件转换为WebSocket消息，供Unity等前端连接
import json
import asyncio
import threading
import logging
from typing import Set, Dict, Any, Optional
from dataclasses import asdict

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    from websockets.exceptions import ConnectionClosed
except ImportError:
    websockets = None
    WebSocketServerProtocol = None

from project.managers import CALLBACK_MANAGER
from project.config.logging_config import setup_logging
from project.config.settings import BRIDGE_ENABLED, BRIDGE_HOST, BRIDGE_PORT
from project.events.event_types import (
    CALIBRATION_START_REQUEST,
    ROUGH_GAZE_UPDATE,
    GAZE_POINT_UPDATE,
    REQUEST_CALIBRATION_UI_CLOSE,
    SYSTEM_STOP,
    CALIBRATION_POINT_SUBMIT,
    CALIBRATION_COMPLETE,
)
from project.data.data_models import CalibrationRequest, CalibrationResponse, Point2D

logger = setup_logging(__name__)


class BridgeServer:
    """WebSocket桥接服务器 - 将事件转换为WebSocket消息"""
    
    def __init__(self, host: str = BRIDGE_HOST, port: int = BRIDGE_PORT):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.running = False
        self.server = None
        self.server_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """注册事件处理器，将事件转换为WebSocket消息"""
        CALLBACK_MANAGER.register(CALIBRATION_START_REQUEST, self._on_calibration_start_request)
        CALLBACK_MANAGER.register(ROUGH_GAZE_UPDATE, self._on_rough_gaze_update)
        CALLBACK_MANAGER.register(GAZE_POINT_UPDATE, self._on_gaze_point_update)
        CALLBACK_MANAGER.register(REQUEST_CALIBRATION_UI_CLOSE, self._on_request_calibration_ui_close)
        CALLBACK_MANAGER.register(SYSTEM_STOP, self._on_system_stop)
        logger.info("桥接服务器事件处理器已注册")
    
    def _cleanup_event_handlers(self):
        """清理事件处理器"""
        CALLBACK_MANAGER.unregister(CALIBRATION_START_REQUEST, self._on_calibration_start_request)
        CALLBACK_MANAGER.unregister(ROUGH_GAZE_UPDATE, self._on_rough_gaze_update)
        CALLBACK_MANAGER.unregister(GAZE_POINT_UPDATE, self._on_gaze_point_update)
        CALLBACK_MANAGER.unregister(REQUEST_CALIBRATION_UI_CLOSE, self._on_request_calibration_ui_close)
        CALLBACK_MANAGER.unregister(SYSTEM_STOP, self._on_system_stop)
        logger.info("桥接服务器事件处理器已清理")
    
    def _serialize_point2d(self, point: Point2D) -> Dict[str, float]:
        """序列化Point2D为字典"""
        return {"x": float(point.x), "y": float(point.y)}
    
    def _create_message(self, event: str, payload: Dict[str, Any]) -> str:
        """创建WebSocket消息（JSON格式）"""
        message = {
            "event": event,
            "payload": payload
        }
        return json.dumps(message, ensure_ascii=False)
    
    async def _broadcast(self, message: str):
        """广播消息给所有连接的客户端"""
        if not self.clients:
            return
        
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"发送消息到客户端失败: {e}", exc_info=True)
                disconnected.add(client)
        
        # 移除断开的客户端
        self.clients -= disconnected
        if disconnected:
            logger.info(f"移除了 {len(disconnected)} 个断开的客户端连接")
    
    def _on_calibration_start_request(self, frame_id: int = None):
        """处理标定启动请求事件"""
        if not self.running or not hasattr(self, 'loop') or self.loop is None:
            return
        
        payload = {}
        if frame_id is not None:
            payload["frame_id"] = frame_id
        
        message = self._create_message(CALIBRATION_START_REQUEST, payload)
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)
    
    def _on_rough_gaze_update(self, calibration_request: CalibrationRequest):
        """处理粗略视线更新事件"""
        if not self.running or not hasattr(self, 'loop') or self.loop is None:
            return
        
        payload = {
            "frame_id": calibration_request.frame_id,
            "eye_type": calibration_request.eye_type,
            "intersection": self._serialize_point2d(calibration_request.intersection)
        }
        message = self._create_message(ROUGH_GAZE_UPDATE, payload)
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)
    
    def _on_gaze_point_update(self, point: Point2D, color: str = "#0000FF"):
        """处理视线点更新事件"""
        if not self.running or not hasattr(self, 'loop') or self.loop is None:
            return
        
        payload = {
            "point": self._serialize_point2d(point),
            "color": color
        }
        message = self._create_message(GAZE_POINT_UPDATE, payload)
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)
    
    def _on_request_calibration_ui_close(self):
        """处理标定UI关闭请求事件"""
        if not self.running or not hasattr(self, 'loop') or self.loop is None:
            return
        
        message = self._create_message(REQUEST_CALIBRATION_UI_CLOSE, {})
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)
    
    def _on_system_stop(self):
        """处理系统停止事件"""
        if not self.running or not hasattr(self, 'loop') or self.loop is None:
            return
        
        message = self._create_message(SYSTEM_STOP, {})
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)
    
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """处理客户端连接"""
        self.clients.add(websocket)
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"新客户端连接: {client_addr} (总连接数: {len(self.clients)})")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    event = data.get("event")
                    payload = data.get("payload", {})
                    
                    # 处理客户端发送的事件
                    await self._handle_client_message(event, payload)
                except json.JSONDecodeError as e:
                    logger.error(f"解析客户端消息失败: {e}, 消息: {message}")
                except Exception as e:
                    logger.error(f"处理客户端消息失败: {e}", exc_info=True)
        
        except ConnectionClosed:
            logger.info(f"客户端断开连接: {client_addr}")
        except Exception as e:
            logger.error(f"客户端连接错误: {e}", exc_info=True)
        finally:
            self.clients.discard(websocket)
            logger.info(f"客户端已移除: {client_addr} (剩余连接数: {len(self.clients)})")
    
    async def _handle_client_message(self, event: str, payload: Dict[str, Any]):
        """处理客户端发送的消息，转换为进程内事件"""
        try:
            if event == CALIBRATION_POINT_SUBMIT:
                # 解析CalibrationResponse
                calibration_response = CalibrationResponse(
                    frame_id=payload.get("frame_id"),
                    eye_type=payload.get("eye_type"),
                    target_pixel=Point2D(
                        x=payload["target_pixel"]["x"],
                        y=payload["target_pixel"]["y"]
                    ),
                    background_color=payload.get("background_color", "black")
                )
                # 发送到进程内事件总线
                CALLBACK_MANAGER.emit(CALIBRATION_POINT_SUBMIT, calibration_response)
                logger.debug(f"转发标定点提交事件: frame_id={calibration_response.frame_id}")
            
            elif event == CALIBRATION_COMPLETE:
                # 解析标定完成数据
                calibration_result = payload.get("calibration_result", {})
                # 发送到进程内事件总线
                CALLBACK_MANAGER.emit(CALIBRATION_COMPLETE, calibration_result)
                logger.debug(f"转发标定完成事件: {calibration_result}")
            
            else:
                logger.warning(f"未知的客户端事件: {event}")
        
        except Exception as e:
            logger.error(f"处理客户端消息失败: {e}", exc_info=True)
    
    async def _run_server(self):
        """运行WebSocket服务器"""
        try:
            async with websockets.serve(self._handle_client, self.host, self.port):
                logger.info(f"WebSocket桥接服务器已启动: ws://{self.host}:{self.port}")
                self.running = True
                # 保持服务器运行
                await asyncio.Future()  # 永远等待
        except Exception as e:
            logger.error(f"WebSocket服务器启动失败: {e}", exc_info=True)
            self.running = False
    
    def _run_in_thread(self):
        """在线程中运行asyncio事件循环"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._run_server())
        except Exception as e:
            logger.error(f"WebSocket服务器运行错误: {e}", exc_info=True)
        finally:
            self.loop.close()
    
    def start(self):
        """启动桥接服务器（在独立线程中）"""
        if websockets is None:
            logger.error("websockets库未安装，无法启动桥接服务器。请运行: pip install websockets")
            return
        
        if self.running:
            logger.warning("桥接服务器已在运行")
            return
        
        self.server_thread = threading.Thread(
            target=self._run_in_thread,
            name="BridgeServerThread",
            daemon=True
        )
        self.server_thread.start()
        logger.info("桥接服务器线程已启动")
    
    def stop(self):
        """停止桥接服务器"""
        if not self.running:
            return
        
        self.running = False
        self._cleanup_event_handlers()
        
        # 关闭所有客户端连接
        if self.loop and not self.loop.is_closed():
            for client in list(self.clients):
                asyncio.run_coroutine_threadsafe(client.close(), self.loop)
        
        # 等待线程结束
        if self.server_thread and self.server_thread.is_alive():
            # 由于使用了Future()，需要取消事件循环
            if self.loop and not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self.loop.stop)
            self.server_thread.join(timeout=2.0)
        
        logger.info("桥接服务器已停止")


def run_bridge_server(host: str = BRIDGE_HOST, port: int = BRIDGE_PORT) -> BridgeServer:
    """创建并返回桥接服务器实例"""
    return BridgeServer(host=host, port=port)
