const gulp = require('gulp'),
  concat = require('gulp-concat'),
  replace = require('gulp-replace'),
  babel = require('gulp-babel'),
  sass = require('gulp-sass'),
  autoprefixer = require('gulp-autoprefixer'),
  cssmin = require('gulp-csso'),
  uglify = require('gulp-uglify'),
  htmlmin = require('gulp-htmlmin'),
  fs = require('fs'),
  pump = require('pump');
const skeletonCssRoot = 'node_modules/skeleton-css/css/';
const requireJsPath = 'node_modules/requirejs/require.js';

gulp.task('styles', function(cb) {
  pump(
    [
      gulp.src([skeletonCssRoot + 'normalize.css', skeletonCssRoot + 'skeleton.css', 'src/scss/main.scss']),
      sass(),
      concat('bundle.css'),
      autoprefixer(),
      cssmin(),
      gulp.dest('dist')
    ],
    cb
  );
});

gulp.task('js', function(cb) {
  pump(
    [
      gulp.src([requireJsPath, 'src/js/main.js']),
      concat('bundle.js'),
      babel({
        presets: ['env']
      }),
      uglify(),
      gulp.dest('dist')
    ],
    cb
  );
});

gulp.task('html', ['styles', 'js'], function(cb) {
  pump(
    [
      gulp.src('src/html/*.html'),
      replace(/<link href="bundle\.css"[^>]*>/, function(s) {
        const style = fs.readFileSync('dist/bundle.css', 'utf-8');
        return '<style>\n' + style + '\n</style>';
      }),
      replace(/<script src="bundle\.js"[^>]*><\/script>/, function(s) {
        const script = fs.readFileSync('dist/bundle.js', 'utf-8');
        return '<script type="text/javascript">\n' + script + '\n</script>';
      }),
      htmlmin({collapseWhitespace: true}),
      gulp.dest('dist'),  // for development preview
      gulp.dest('../templates')  // for publishing as template
    ],
    cb
  );
});

gulp.task('default', ['html']);
