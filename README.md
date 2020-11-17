# Data Science Portfolio Website

This is my portfolio and blog site which can be reached @ https://salihkilicli.github.io/. 

![](https://github.com/salihkilicli/salihkilicli.github.io/blob/master/ss.png)

The webpage was built on Hugo's [Toha](https://themes.gohugo.io/toha/) theme and the creator of the theme is [Emruz Hossain](https://github.com/hossainemruz/). 

The avatar is created using **avatarmaker**, you can create your own avatar using https://avatarmaker.com/.


Notes:

- I will be sharing blog posts related to Mathematics, Statistics and Data Science under the `Posts` section.
- I will also be sharing my personal Data Science projects under the `Projects` section.
- Under the `Notes` section I will be posting my personal notes from my favorite books & MOOC about Data Science/ML/DL as well as from scratch tutorials.

#### How to add particles effects on a section in your website?

Here is a **preview** of how the particles effects works:

![](https://github.com/salihkilicli/salihkilicli.github.io/blob/master/website.gif)

#### Details: ####

There are so many tutorials on how to use particle effects on the web; however, almost all of them create a new HTML file rather than applying it onto an existing website with a fixed background. I wanted to explain how to do it in this post. 

Here is the GitHub repo for the [particles.js](https://github.com/VincentGarreau/particles.js/).
You can edit and preview the live version of the effect on [here](https://vincentgarreau.com/particles.js/), and pick best parameters for your own animation.
This [website](https://codepen.io/pen/?&editable=true=https%3A%2F%2Fvincentgarreau.com%2Fparticles.js%2F) provides HTML, CSS and js codes you want to include in your website to add `particles` effect in your webpage.

In order to add particles effect on your website, follow the instructions below:

1) Download the demo files from the **repo**.
2) Copy `particles.js` and `app.js` file into your local `js` folder.
3) Edit your webpage's `index.html` file and insert a `particles.js` container (or attribute) inside the section you want to apply the effect.

  ```html
  <!-- particles-js container -->
  <div id="particles-js"></div>
  ```
4) Then, using a `<script>` tag include the relative path to `particles.js` and `app.js` files.

  ```html
    <!-- Particles Animation JS -->
    <script src="assets/js/particles.js"></script>
    <script src="assets/js/particles.app.js"></script>
  ```
5) Now insert the __CSS__ code below inside the corresponding section's (in my case it was `home.css`) css file:

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
