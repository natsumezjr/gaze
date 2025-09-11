module.exports = {
  publicPath: './',
  outputDir: 'dist',
  assetsDir: 'assets',
  productionSourceMap: false,
  // 解决页面缩放问题的配置
  chainWebpack: config => {
    config.plugin('html')
      .tap(args => {
        args[0].meta = {
          viewport: 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no'
        };
        return args;
      });
  }
};