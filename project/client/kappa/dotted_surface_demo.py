"""动态点阵表面演示程序"""
import tkinter as tk
from tkinter import ttk
from project.client.kappa.dotted_surface import DottedSurface


def create_demo_window():
    """创建演示窗口"""
    root = tk.Tk()
    root.title("动态点阵表面演示")
    
    # 获取屏幕尺寸并设置为全屏
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}")
    
    # 创建点阵表面（暗色主题）
    surface = DottedSurface(
        root,
        theme='dark',
        bg_color='#000000'
    )
    surface.pack(fill=tk.BOTH, expand=True)
    
    # 创建覆盖层用于显示文字（可选）
    overlay_frame = tk.Frame(
        root,
        bg='#000000',
        highlightthickness=0
    )
    overlay_frame.pack(fill=tk.BOTH, expand=True)
    
    # 创建文字标签（居中显示）
    text_label = tk.Label(
        overlay_frame,
        text="Dotted Surface",
        font=("Courier", 48, "bold"),
        fg="#FFFFFF",
        bg="#000000"
    )
    text_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    # 创建控制面板（可选）
    control_frame = tk.Frame(
        root,
        bg='#1a1a1a',
        relief=tk.FLAT
    )
    control_frame.place(x=20, y=20)
    
    # 主题切换按钮
    def toggle_theme():
        current_theme = surface.theme
        new_theme = 'light' if current_theme == 'dark' else 'dark'
        surface.set_theme(new_theme)
        
        # 更新文字颜色
        if new_theme == 'dark':
            text_label.config(fg="#FFFFFF", bg="#000000")
            overlay_frame.config(bg="#000000")
        else:
            text_label.config(fg="#000000", bg="#FFFFFF")
            overlay_frame.config(bg="#FFFFFF")
        
        theme_btn.config(text=f"主题: {new_theme}")
    
    theme_btn = tk.Button(
        control_frame,
        text="主题: dark",
        command=toggle_theme,
        bg='#333333',
        fg='#FFFFFF',
        relief=tk.FLAT,
        padx=10,
        pady=5,
        font=("Arial", 10)
    )
    theme_btn.pack(side=tk.LEFT, padx=5)
    
    # 停止/启动动画按钮
    def toggle_animation():
        if surface.is_running:
            surface.stop_animation()
            anim_btn.config(text="启动动画")
        else:
            surface.start_animation()
            anim_btn.config(text="停止动画")
    
    anim_btn = tk.Button(
        control_frame,
        text="停止动画",
        command=toggle_animation,
        bg='#333333',
        fg='#FFFFFF',
        relief=tk.FLAT,
        padx=10,
        pady=5,
        font=("Arial", 10)
    )
    anim_btn.pack(side=tk.LEFT, padx=5)
    
    # 窗口关闭事件
    def on_closing():
        surface.stop_animation()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    return root


def main():
    """主函数"""
    root = create_demo_window()
    root.mainloop()


if __name__ == "__main__":
    main()
