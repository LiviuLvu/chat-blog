---
title: "Failing to sell a build optimization idea and not giving up"
date: "2018-11-25"
summary: "Have you ever tried to explain something that could be very helpful but seems too complicated to others because you did not properly present your idea?"
# description: ""
tags: ["Git", "Bash", "Teamwork", "Process Optimization", "Scripting"]
author: "Liviu Iancu"
weight: 3
series: ["Automation"]
# categories: [""]
# aliases: [""]
# cover:
#   image: images/img.png
#   caption: "Description [text](https://link.somewhere/)"
---
Have you ever tried to explain something that could be very helpful but seems too complicated to others because you did not properly present your idea?

In this article I want to tell a story about my mistake in proposing a (kind of complicated) solution to optimize the build process for the front end team I am part of. I will have to leave out a lot of juicy details about the project because it’s still ongoing work.

### This is the plot
We have a pretty nasty build that can take more than 2 minutes to complete. If not restarted often, after each change, this time bending monster can grow to 4 minutes or more. Then send the css through CORS to a remote machine. There are other limitations involving security and limited access to part of the code base.

Imagine yourself try to style a few tabs or for any UI work your heart desires and then waiting like a monk to see those changes in the browser.

In this build recipe we throw 4 theme color files (among other tech). One of them is the old version that we are working hard to replace in the hopefully near future (its a big whale of a project). At this time we only need 2 of these themes.

The themes are developed using sass files and besides their processing there are also some internationalization files and other javascript files that we don’t touch in our css beautifying process.

### The idea
Many moons ago (few months back) we had 2 themes and the build was “only” 30 seconds. Now we have 4. So until we finalize the layout and color variables why not keep the 2 new themes out of the build?

Oh, how wonderful life could be… Well, kind of: 30 seconds is not that great but is much better than 2+ minutes to wait for a darn padding change.

### We have to do something
In order to achieve this outstanding tech innovation (temporary remove files from the build) we first discussed it in a meeting with “higher order council”. I think we got a “we will look into it” response but removing the 2 extra themes was off the menu.

Meanwhile we keep adding css and the build time keeps growing. None of us on the front end team really understands all the build settings involving .sh script files that connect our color themes and other contraptions that our project is connected to (…long story). We get by with testing css changes in dev-tools and copying them in the project files.

Being a curious fella, I started looking into how the theme files are connected with the .sh script files. After poking around (at this time bash scripting is too gibberish for me) i hacked a way to remove 2 themes and shorten build and watcher update times.

Jolly good, good times ahead, but wait, I have to share this change with front end team and we are not allowed to push any of these changes to main branch.

To get around the updating branch restriction, I did some `git ignore` and `git update-index — assume-unchanged` commands to make git ignore changes to the removed theme files. In my opinion not that many or difficult changes but I was so wrong.

This is where I was reminded that people need simple solutions to understand the concept and take action.

### I’m not a very good sales man
Actually I probably suck at selling because the sharing plan did not go well and the optimization was received as complicated.

Shorter build time - useful stuff, right? I had a go at letting everybody know I am working on shenanigans that would make our build faster and everyone was interested.

So there I go, building this wonderful instructions doc (the kind of user manual nobody reads), with screen shots and git commands (a painting of rotten apples might have been better received).

The magic vanished when the guys and gals visited the instructions doc. I was expecting resistance so I went and simplified a lot of the file and removed as much text and images as possible while keeping the essential. After announcing the upgrade, based on feedback no one asked any questions and it was clear I failed to present a usable, practical solution.

### One more try
Only one hope remains to make the build optimization steps look attractive (more like push this button instead of read this user manual). If I can use bash scripting to automate all those changes in one command then it just might work for everyone.

If you have a similar story please share.

To be continued.