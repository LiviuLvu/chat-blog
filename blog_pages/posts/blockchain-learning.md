---
title: "Learning blockchain programming"
date: "2022-03-03"
summary: "In the last couple of months I became curious about what makes a crypto currency work, and why I see more and more news about the technology behind it, promoting it as the future solution to so many problems.
How does blockchain work?"
# description: ""
tags: ["Blockchain", "Truffle", "Learning"]
author: "Liviu Iancu"
weight: 5
series: ["Programming"]
# categories: [""]
# aliases: [""]
# cover:
#   image: images/img.png
#   caption: "Description [text](https://link.somewhere/)"
---
In the last couple of months I became curious about what makes a crypto currency work, and why I see more and more news about the technology behind it, promoting it as the future solution to so many problems.
How does blockchain work?

One of my paths in blockchain land went to, what I thought would be an easy start (I was a tad wrong). Learning the basics by trying out the truffle tutorial “pet-shop”.

https://trufflesuite.com/tutorial

If you thought the frontend environment was changing fast and there are a lot of frameworks popping up every week, wait till you get your fingers dirty writing some blockchain code.

### Node version troubles

With versions. Thank god that nvm exists. Tried node 8, 10, 12, 14, 16 then npm install, install issue, something about the contracts was just not working as expected…

Tried a bunch of solidity compiler versions until one of them compiled the damn contracts without the issue: ParserError: Source file requires different compiler version.

When you see a command like `npm install -g truffle` you (or maybe you do) but I did not realize the tutorial is a bit old and that install command will use the latest truffle version. Some more headache for me, yaay!

I switched versions until some more neurons died and the remaining ones realized the truffle and solidity compiler versions needed to match with the old tutorial.

The final combination that worked then was:

truffle@5.2.0 | solidity compiler v0.5.16 | node v16 (older versions of node worked in the end, after the compiler and truffle versions were compatible with the contracts)

Don’t expect it to work now when you are reading this :)

This is the code base after installing everything mentioned in the tutorial:  
https://github.com/LiviuLvu/etherum-solidity-pet-shop

Aditionally, to the tutorial code, I added some node packages like Prettier and Husky to format the code, because I think it’s a chore to format the code manually, in a consistent way, so I like to have this done automatically on save and before git commit hook.

Pro tip from:  
https://trufflesuite.com/guides/debugger-variable-inspection/#debugging-errors  
<em>
“Sometimes, contracts have errors in them. They may compile just fine, but when you go to use them, problems arise.  
This is one of the reasons to use Ganache over a public testnet: you can easily redeploy a contract without any penalty in time or ether.”
</em>

### Eventually
All the parts were working together. From the local repo the deploy to Ganache local network worked.

The Ganache UI is very nice and after deploying a contract or adopting a pet, was showing a lot of new info that I did not understand at first.

The Metamask wallet was connected to Ganache and was showing the change in the amount of coin when adopting a pet.

### Overall

Going through the Truffle tutorial was worth it as a quick way to get a general idea of the Ethereum environment. Totally recommend it, even with the difficulties with versions mentioned above.

For the following experiments I am looking at Remix, or try out the environment from Hardhat. There are a lot of good views about their tools.

Maybe after trying the Hardhat tutorial it would be nice to write a comparison article with Truffle?

Until then, have fun learning!