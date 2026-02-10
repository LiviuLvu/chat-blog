---
title: "Automate tasks and share using script files"
date: "2018-12-30"
summary: "A bit about using the power of scripts to help my comrades automate repeated tasks and process optimizations."
# description: ""
tags: ["Bash", "Scripting", "Sharing", "Node", "Automate"]
author: "Liviu Iancu"
weight: 5
series: ["Automation"]
# categories: [""]
# aliases: [""]
# cover:
#   image: images/img.png
#   caption: "Description [text](https://link.somewhere/)"
---
In this article I write a bit about using the power of scripts to help my comrades automate repeated tasks and process optimizations.

This is a follow up to previous article, where i wrote about a fail that gave me the motivation to learn bash scripting. I tried to do some build optimizations manually and they worked but there were too many steps to be a shareable solution.

### Make it friendly
When you share a script with a fellow programmer, put some friendly [ASCII art](https://medium.com/r/?url=https%3A%2F%2Fasciiart.website%2Findex.php%3Fart%3Danimals%2Fhorses) in there so it will appear in the command line to signal the end of the execution. It makes things more lively and fun.

__I placed an elephant in a script file, shared with team and wrote them that all is well if they see an elephant at the end of the process.__

Sharing a bunch of steps automated in a script file was my starting motivation.  
Learning to code in bash was useful and fun but very intimidating at first. After getting many syntax errors and sometimes worrying that I will break things in irreparable ways I got the basics and started automating whatever I could. Below are solutions shared at work.

### Automation ideas
Here are some ideas we tried and made our tasks flow more like the ki energy in a zen garden:

Build optimization to temporarily modify some files that we were not allowed to modify (not solvable with git ignore).  
Morning grind routine when we needed to re-connect our local dev environment to a virtual remote machine.  
Update a branch with latest changes while still working on it (a bunch of git commands mashed together in one script).  
Automate daily ssh commands where the script autocompletes command line questions where needed (ex. login details).  
The best part of making and sharing a script was that it sparked curiosity about the possibilities of automation and how we could use it in our daily workflow.  

### Fun Fact
Did you know there are [games](https://medium.com/r/?url=https%3A%2F%2Fwww.tecmint.com%2Fbest-linux-terminal-console-games%2F) you can play in the command line?  

### More to come: Node.js

While searching for automation ideas for my workflow i discovered that using Node instead of bash scripting could be even more rewording as there are many packages on NPM to choose from and node is cross platform.

Found great book on the subject of automating stuff using node: [Automating with Node.js](https://medium.com/r/?url=https%3A%2F%2Fleanpub.com%2Fautomatingwithnodejs) by Shaun Michael Stone. Book goes on to do list!

The reword for succeeding after a fail and helping others is treasure for the soul. I hope this article inspires you to explore, discover and improve your every day.