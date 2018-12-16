// Setup require.js
const jsPaths = Object.assign(
  {
    'jquery': 'https://cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min',
    'mathjax': 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS_HTML',
  },
  window.jsPaths || {}
);
requirejs.config({
  "paths": jsPaths
});
