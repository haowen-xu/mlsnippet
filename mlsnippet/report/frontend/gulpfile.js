const gulp = require('gulp'),
  concat = require('gulp-concat'),
  replace = require('gulp-replace'),
  babel = require('gulp-babel'),
  sass = require('gulp-sass'),
  postcss = require('gulp-postcss'),
  uglify = require('gulp-uglify'),
  htmlmin = require('gulp-htmlmin'),
  fs = require('fs'),
  pump = require('pump');
const requireJsPath = 'node_modules/requirejs/require.js';

gulp.task('sass-preprocessor', function(cb) {
  pump(
    [
      gulp.src('src/scss/*.scss'),
      // PhpStorm requires @import "~..." CSS from node_modules for code completion,
      // but gulp-sass does not support it.  So we hack it.
      replace(/^@import (["'])~(.*?)\1/gm, '@import "$2"'),
      gulp.dest('dist/scss')
    ],
    cb
  );
});

gulp.task('css', ['sass-preprocessor'], function(cb) {
  const includePaths = [ 'dist/scss', 'node_modules' ];
  const autoprefixer = require('autoprefixer'),
    cssnano = require('cssnano'),
    atImport = require('postcss-import');
  const postCssPlugins = [
    autoprefixer(),
    atImport({ path: includePaths }),
    cssnano()
  ];
  pump(
    [
      gulp.src('dist/scss/*.scss'),
      sass({
        includePaths: includePaths,
      }),
      postcss(postCssPlugins),
      gulp.dest('dist')
    ],
    cb
  );
});

gulp.task('ipython-css', ['css'], function(cb) {
  pump(
    [
      gulp.src('dist/ipython.css'),
      gulp.dest('../templates')
    ]
  )
});

gulp.task('js', function(cb) {
  pump(
    [
      gulp.src([requireJsPath, 'src/js/main.js']),
      concat('main.js'),
      babel({
        presets: ['env']
      }),
      uglify(),
      gulp.dest('dist')
    ],
    cb
  );
});

gulp.task('html', ['css', 'js'], function(cb) {
  pump(
    [
      gulp.src('src/html/*.html'),
      replace(/<link href="bundle\.css"[^>]*>/, function(s) {
        const style = fs.readFileSync('dist/main.css', 'utf-8');
        return '<style>\n' + style + '\n</style>';
      }),
      replace(/<script src="bundle\.js"[^>]*><\/script>/, function(s) {
        const script = fs.readFileSync('dist/main.js', 'utf-8');
        return '<script type="text/javascript">\n' + script + '\n</script>';
      }),
      htmlmin({collapseWhitespace: true}),
      gulp.dest('dist'),  // for development preview
      gulp.dest('../templates')  // for publishing as template
    ],
    cb
  );
});

gulp.task('default', ['html', 'ipython-css']);
