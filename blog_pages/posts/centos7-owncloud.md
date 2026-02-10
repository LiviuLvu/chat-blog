---
title: "Setting up a home CentOS 7 server with OwnCloud"
date: "2018-07-15"
summary: "Lessons from High Performance Browser Networking from Ilya Grigorik. Fantastic resource, great help to understand how networks work."
# description: ""
tags: ["Linux", "Centos7", "Home Server", "Owncloud", "Networks"]
author: "Liviu Iancu"
weight: 2
series: ["Network"]
# categories: [""]
# aliases: [""]
# cover:
#   image: images/img.png
#   caption: "Description [text](https://link.somewhere/)"
---
A few weeks ago I got myself a little desktop that can serve as a home server… some 16Gb DDRam3, 2 SSD 60Gb each, 2 Tb HDD + 1Tb HDD… hmm, lots of serving possibilities :)

Mission? Unclear .. wait, a file server would be nice to have.

Actually the main reason for this headache is to learn about servers and networks, because in frontend (my job) there were many situations where I was confused by errors involving network requests, errors from server, network and so on.

Used the word headache because that is what it turned into, but man, I learned a lot about computer networks and working in the command line linux. Which reminds me of…

### A great book on networks
Oh, and I started reading from High Performance Browser Networking from Ilya Grigorik. Fantastic resource, great help to understand how networks work. There are so many things that can go wrong on the data transfer path.
https://hpbn.co/#toc

While searching for tools and protocols to share files I found OwnCloud and proceeded to install all the required packages including the kitchen sink and then uninstall, clean and reinstall some packages a couple of times :) Errors were attacking from every direction.

As a new comer to Linux and servers, saying it was hard is an understatement.
After gatering a lot of documentation I figured it would be worth while to write an article about it.

These are some of the commands i encountered:

Centos7 useful commands
nmcli > command line tool for controlling network manager  
nmtui > text ui for controlling network manager  
nmap > Network exploration tool and security  
lsof > list open files  
netstat > Print network connections, routing tables, interface   statistics, masquerade connections, and multicast memberships
ifconfig > configure a network interface  
ip > show / manipulate routing, devices, policy routing and tunnels  
lshw > list hardware  
ip addr show > check the current IP address  
ip r > show default gateway  
hostnamectl > display host name + info  
yum install links > “Links” command line web browser (this was fun)  
sudo top > list proceses running on server  
ip addr show eno1 | grep inet | awk ‘{ print $2; }’ | sed ‘s/\/.*$//’ > shows public ip address  
nslookup yourhost.yourdomain.com > Is the DNS server returning the   right IP ?

netstat -anp | grep 443 > what is using port 443  
fuser 443/tcp > what is using port 443  
netstat -lnp | grep 443 > what is using port 443  
kill -9 #### > stop process number: #### using port 443  
ss > is a useful power tool to inspect various statistics for open sockets  
ss — options — extended — memory — processes — info > to see the current peers and their respective connection settings  

Package Groups options instructions https://www.certdepot.net/rhel7-get-started-package-groups/
About PHP server test and details
nano /var/www/html/info.php
https://localhost/info.php > File should be deleted after test

rpm -qa | grep -i [RPM_PACKAGE_name] > search for rpm package
rpm -e [RPM_PACKAGE_name] > delete it

### Server setup
My server setup currently looks like this:
CentOS 7
Own Cloud
Apache server
MariaDB

Finally got this contraption to start. SoundCloud is merrily serving files but only inside LAN.
After doing some port forwarding and static IP asignment shenanigans inside the router and following Centos settings tutorials, Apache is able to serve files outside LAN through DDNS servers from noip.com

There is much more to be learned, especially security stuff, but I will stop for now. This rabbit hole is getting deep and I also have a lot of frontend subjects to experiment.

Have fun and keep exploring!