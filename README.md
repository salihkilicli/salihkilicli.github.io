# Data Science Portfolio Website

This is my portfolio and blog site which can be reached @ https://salihkilicli.github.io/

The webpage was built on Hugo's [Toha](https://themes.gohugo.io/toha/) theme and the creator of the theme is [Emruz Hossain](https://github.com/hossainemruz/).

Notes:

- I will be sharing blog posts related to Mathematics, Statistics and Data Science under the `Posts` section.
- I will also be sharing my personal Data Science projects under the `Projects` setion.
- Under the `Notes` section I will be posting Data Science/ML/DL tutorials (from scratch) and notes to remind myself some of the advanced topics I don't utilize regularly.

#### How to add particles effects on a section in your website?

There are so many tutorials on how to use particle affects on the web; however, almost all of them create a new HTML file rather than applying it onto an existing website with fixed background. I wanted to explain how to do it in this post. 

Here is the GitHub repo for the particles.js https://github.com/VincentGarreau/particles.js/.
You can edit and check live version of the effect on [here](https://vincentgarreau.com/particles.js/), and pick best parameters for your own particles.
This [website](https://codepen.io/pen/?&editable=true=https%3A%2F%2Fvincentgarreau.com%2Fparticles.js%2F) allows you to see HTML, CSS and js codes you want to include in your website to add `particles` effect in your webpage.

In order to complete adding particles effect in your website follow the instructions below:

1) Download the demo files from the repo.
2) Copy `particles.js` and `app.js` file into your local `js` folder.
3) Edit your webpages `index.html` file and insert a `particles.js` container (or attribute) inside the section you want to apply it in.

  ```html
  <!-- particles-js container -->
  <div id="particles-js"></div>
  ```
4) Also, using `<script>` tag include relative directions to `particles.js` and `app.js` files you want to apply in your webpage. Notice the path given below is relative to the folder `index.html` belongs.

  ```html
    <!-- Particles Animation JS -->
    <script src="assets/js/particles.js"></script>
    <script src="assets/js/particles.app.js"></script>
  ```
5) Now insert the __js__ code below inside the corresponding section's (in my case it was `home.css`) css file:

  ```javascript
  .particles {
  margin: 0px !important;
  top:    0px !important;
  left:   0px !important;
  right:  0px !important;
  bottom: 0px !important;
  z-index: -1;
  }
  ```
