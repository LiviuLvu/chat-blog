---
title: "Comparing four ai generators generating a voting app"
date: "2025-10-11"
summary: "Using the same prompt on four ai app generators to see how they compare. Main goal is to generate a voting web app"
# description: ""
tags: ["ai", "prompting", "automation"]
author: "Liviu Iancu"
weight: 1
series: ["Automation"]
# categories: [""]
# aliases: [""]
# cover:
#   image: images/img.png
#   caption: "Description [text](https://link.somewhere/)"
---
Using the same prompt on four ai app generators to see how they compare. Main goal is to generate a voting web app.

## Tested same prompt on 4 app generators

Build a voting app that sorts a list on the votes received. The user is only allowed to vote after being logged in.  
Do not code just chat with me  
Do you have any clarifying questions that would help you deploy this request without bugs?  
Answer me in great detail. Why do you think this will work  
Let me know if you understand what the task is before making edits  
Tell me what you are going to do step-by-step and wait for my approval  

The prompt is mostly inspired by this Reddit:  
https://www.reddit.com/r/ChatGPTPromptGenius/comments/1il4ias/i_built_over_20_apps_using_ai_tools_these_are_my/

## Results
Strangely most clarification questions asked by AI looked remarkably similar across the services. I did not really check who used what AI provider under the hood. This are the pro and cons i observed. The best results i had were from Lovable, but it depends highly on the requirements. You have to try yourself.

## Lovable
Tech observed in source files: typescript, vite, tailwind, supabase

+beautiful design, centered minimalist ui  
+sorting works nice based on votes and newest  
+notifications popup showing forbidden action  
+log in/out works  
+nice overview of db with users and items, appeared hashed  
+nice security scan that suggests fixes and docummentation-sign in required no mail confirm, fake  
-critical file .env containing DB details was not added to git ignore  


**Conclusion**  
Happy days, not exactly what i wanted but good experience, good design over all, everything works as expected.

## V0 ( vercel )
Tech observed in source files: react js, tailwind, supabase

+good clarifying questions, responded with about 6 detailed clarification points  
+sign in with google oauth sent confirm mail, appeared to work (only on first generated app)  
+overall smooth experience and good design  
-routing not working after login, had to guess route  
-did not find option to choose FE tech for generated results  
-nothing updates without refresh  
-voting counter jumps between 1 and -1  
-could not find where to see the contents of db   
-logs were not showing  
-generate is cumbersome, requires multiple click steps with wait between  
-after trying to generate same app a second time it went a lot worse:  
-generation stopped, i had to prompt wake it up, ended in some unhandled promise err, prompted again, login btn not reacting  

**Conclusion**  
Good design, many bugs. Login appeared to work the first time, but after many ui bugs i tried generating the app a second time and now the login button was not reacting at all.  
Pointed the problem to the bot 2 times and each time the ai confidently replied it is "fixed" but the button was still not reacting.

## Replit
Tech observed in source files: vite, tailwind, radix-ui, postgresql?

+creates check points without asking, nice  
+very nice design and sample datao  
-no route change when logged in / out  
-even tough i specified a login screen as landing page it displays content that should have been behind login  
-if you want to test a basic real auth, seems only option is to paywall for implementing "functionality"  
-not sure if any db was added since i only tested the free access  

**Conclusion**  
Good basic design but the UX result was far from what I specified in clarification feedback (questions asked by ai). Navigation was non existant and content was displayed without the user being logged in.

## Flatlogic
Tech mentioned: React, Node, tailwind, o3-mini model?

+docker files generated  
+swagger api 
+nice additional recommended features to my initial prompt based on followup questions  
-lots of questions, got tired before generating anything  
-takes long time to show anything  
-slow ui, cant really inspect the whole code in free mode access  
-generated a blank screen ... no, just really slow start  
-cant sign in a new user  
-whole site seems buggy, i deleted my account and the website started refreshing repeatedly  

**Conclusion**  
The whole site was really slow, not just the generated app. Inspecting the whole code is not possible in the free tier. The ai asked so many clarification questions that I got tired and bored. In the end design was acceptable, but totally non functional buttons.  


## Some conclusions, observations and especially warnings

Danger: If you publish the code on github, in a public repo, there is high risk of exposing sensitive access tokens to your db or other integrated services. In some of the services the .env file was not added to .gitignore

Debugging ... The machine does solve some small bugs fast, but when it fails you either enter an infinite prompting loop or you start learning the craft. 
Would rather download the source after its generated to an acceptable layout and continue in local env.

Productivity ... i am a skeptic. It might be fun first, but after a while you notice the many, subtle ways, it fails with confidence, and needs careful review every time.
For example: Ai outright forgets or ignores instruction like "add 6 sample items in database", even when it asked if you want samples and you confirmed. After a while it asks again.

All negatives aside, it is spectacular what these ai generators were able to achieve with just one ambiguous prompt and 5 to 7 clarification followup questions. It allows for a lot of creativity, exploring ideas in multiple languages with tech stacks you never worked before.

Great tool if you know what you are doing, horrible giant foot gun if you dont' know anything about programming, secure communication between systems and trust the generated code with your wallet.