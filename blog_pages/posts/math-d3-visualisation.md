---
title: "Mathematical Pattern Visualization with D3"
date: "2018-03-30"
summary: "Visualising patterns using the D3 library."
# description: ""
tags: ["JavaScript", "Pattern Generator", "Visualization"]
author: "Liviu Iancu"
weight: 2
series: ["Programming"]
# categories: [""]
# aliases: [""]
# cover:
#   image: images/img.png
#   caption: "Description [text](https://link.somewhere/)"
---
There is this book I have on my desk called “Brainmatics Logic Puzzles” from Ivan Moscovich. A few days ago I opened it on the page about Kaprekar’s Digitadition pattern. That looked very interesting and made a few lines in Illustrator to see the pattern.  
After drawing a few lines I decided to generate the pattern as large as possible and see what happens with large numbers.

This digitadition pattern made by selecting a number and adding to it the sum of its digits.

Example:
take number 25, add 2+5, you get 28, and repeat
28, 38, 49, 62, 70, 77, 91, 101, 103, 107, 115

### View live demo below
Live demo on github: Link to live demo  
https://liviulvu.github.io/d3-kaprekar-pattern/  
Get the code made for this demo here: Code Files  
https://liviulvu.github.io/d3-kaprekar-pattern/  

The interface is zoomable and scrollable thanks to D3 but I did not take the time to fix some UI quirks. For example you can scroll off the screen and there is no reset button to center back the shapes.

The pattern looks a bit like data embedded on a tape, so I searched the web to find ways to convert this pattern to sound. Found two resources for this:
One is Photosounder (Windows), which is a software that can generate sound from images.  
http://photosounder.com/  
The second is a website that can generate sound from number patterns, called MusicAlgorithms.  
http://www.musicalgorithms.org/3.2/  
Ok, this was fun, but cant think of anything useful coming out of this experiment. Moving on.