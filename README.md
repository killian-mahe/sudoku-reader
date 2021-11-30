<!-- PROJECT LOG -->
<br />
<p align="center">
  
<h3 align="center">Sudoku Reader</h3>

  <p align="center">
    An application that is able to read and solve a Sudoku thantks to machine learning.
    <br />
    <a href="https://github.com/killian-mahe/sudoku-reader"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/killian-mahe/sudoku-reader/issues">Report Bug</a>
    ·
    <a href="https://github.com/killian-mahe/sudoku-reader/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

The goal of this project is to build an application to read a sudoku puzzle from a picture and solve it using CSP (Constraint Satisfaction Problems).

### Built With

* [Python 3.9](https://www.python.org/)
* [PySide6](https://pypi.org/project/PySide6/)


<!-- GETTING STARTED -->
## Getting Started

This project is based on Python. No framework is needed, but you must install the PySide6 package.
We strongly suggest your to create a virtual environment to create the app environment.
You can also use the Dockerfile to use with Docker.

### Installation

Before clonning the repository, you should create a new virtual environment (using `venv` for exemple).

1. Clone the project (or use the src folder)
   ```sh
   git clone https://github.com/killian-mahe/sudoku-reader.git
   ```
2. Install the virtual environment
   ```sh
   cd sudoku-reader && python3 -m venv env
   ```
3. Activate the new environment (on Linux)
   ```sh
   source env/bin/activate
   ```
4. Activate the new environment (on Windows with powershell)
   ```sh
   env\Scripts\Activate
   ```
4. Activate the new environment (on Windows with CMD)
   ```sh
   env\Scripts\activate.bat
   ```
5. Install required packages
   ```sh
   pip3 install -r requirements.txt
   ```

### Use

1. Start the project
    ```sh
    python sudoku_reader
    ```

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/killian-mahe/sudoku-reader/issues) for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Killian Mahé - [@killian-mahe](https://www.linkedin.com/in/killian-mah%C3%A9-246928135/) - killianmahe.pro@gmail.com

Project Link: [https://github.com/killian-mahe/sudoku-reader](https://github.com/killian-mahe/sudoku-reader)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
