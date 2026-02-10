---
title: "Trying out the Ergo blockchain environment"
date: "2022-05-08"
summary: "Started with this page https://docs.ergoplatform.com/node/install/ to install a ergo node but it quickly turned into 2 days of searching for alternatives and tutorials."
# description: ""
tags: ["Ergo", "Blockchain", "Learning", "Starting Ergo Node", "Ubuntu"]
author: "Liviu Iancu"
weight: 5
series: ["Programming"]
# categories: [""]
# aliases: [""]
# cover:
#   image: images/img.png
#   caption: "Description [text](https://link.somewhere/)"
---
Starting a local node without any java experience

The documentation looks pretty good at first sight but i would not call it beginner friendly.

Started with this page https://docs.ergoplatform.com/node/install/ to install a ergo node but it quickly turned into 2 days of searching for alternatives and tutorials.

### Nothing worked as easy as described in the docs

1. The long command on the install page to start the node needs to be adjusted, wait, it was adjusted while I was writing this article :)
The file name should be exactly as the jar file you download but it was not indicated in the command with <> or described in instructions.

2. No downloaded file worked, tried 5 times with different jar files. Each time got the corrupt jar file “error: Invalid or corrupt jarfile ergo-…..jar”

3. The script to start a node was infinitely loading with message: “Executing ergo node to obtain API key hash ……….”

4. I tried to change the java version (on some discussions it was a case that solved the corrupted file error)

### Changing the java version was quite trippy, as in tripping over every step.

Installing v11 on ubuntu is really easy, any version is really easy to install but changing the version is another story. For 2 hours i was fighting with all my will power to change active version 11 to newly installed version 17. No command or restart of terminal did any difference on java --version result. Eventually manually deleting the v11 folder finally made the difference … moving on.

Some things in the documentation remain to be interpreted based on experience with Java and Scala which in my case was close to zero.

After 2 days i gave up and it was another week until i had time and patience to try again.
Its a bit confusing but there are 2 pages with similar, but not the same set of instructions for starting a node so i sort of made a trial and error with both:
1 https://docs.ergoplatform.com/node/install/faq/#running-the-node

2 https://github.com/ergoplatform/ergo/wiki/Set-up-a-full-node#running-the-node-for-the-first-time
The sections prerequisites and running the node are of interest for this experiment.

### These are the steps that worked to get a node going on Ubuntu OS

1. Download the ergo repository from Github:
```
git clone https://github.com/ergoplatform/ergo.git
```
Change directory inside that folder:
```
cd your/downloaded/ergo/folder
```
2. Install the scala build tool: https://www.scala-sbt.org/download.html and compile the node

Check installation running command:
```
sbt --version
```
Result output should be similar to:
```
sbt version in this project: 1.6.2

sbt script version: 1.6.2
```
Compile the ergo jar file with command:
```
sbt assembly
```
3. Run the ergo node.

Create a new folder. For my example named ergo-node (creative right…) Navigate to the new folder:
```
cd ~/ergo-node
```
Create new file ergo.conf as instructed in the running node section in the docs: https://docs.ergoplatform.com/node/install/faq/#running-the-node

Start the node using the generated jar file when you followed step 2:
```
java -jar -Xmx4G /your-downloade-ergo-folder/ergo/target/scala-2.12/ergo-<your-version>.jar — mainnet -c ergo.conf
```
After compiling the jar file the command for starting the node finally worked and did not throw any corrupted .jar file error.
Dont’ remember why I created a separate folder where to run the node but this proved useful for running the node. The logs are saved there and some new folders and files are created.

### Off topic: accidentally stopped the node sync 4 times

The interrupt command in the Ubuntu terminal by default is ctrl + c, exactly what is used for copy paste text in other windows.
Accidentally stopped the node sync, by attempting to copy parts of the logs which kept displaying errors similar to: failed to connect to <url>, connection refused or connection timeout.

To avoid accidentally stopping any running processes, the command to change the default copy shortcut ctrl + c interrupt command in Ubuntu terminal is this:
```
stty intr ^i
```
The sync of the node seems to take a lot of hours or days depending on network connection. Cant’ wait for the full sync to start experimenting some more with the api.

I hope this posts saves you some head pills and sleep hours.
Let me know if its useful for you and if you have anything to add or spotted any errors, please leave a comment to guide new comers on the correct path.

Have fun experimenting!
