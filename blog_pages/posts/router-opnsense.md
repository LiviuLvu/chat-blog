---
title: "Building My Own Firewall Router with OPNsense: A Journey in Home Network Security"
date: "2025-01-23"
summary: "Some practical benefits running OPNsense on your home network. It vastly improves safety for anyone at home using the internet."
# description: ""
tags: ["Opnsense", "Home Network", "Router", "Firewall"]
author: "Liviu Iancu"
weight: 1
series: ["Network"]
# categories: [""]
# aliases: [""]
# cover:
#   image: images/img.png
#   caption: "Description [text](https://link.somewhere/)"
---

Never imagined I would be writing about this. I built my own firewall / router using a small form factor PC with OPNsense open source firewall router. The previous "normal" router is now only used as wireless access to OPNsense.

### Why did I do it? 

Its incredibly useful, practical and fun to learn something I am very interested in: blue team, SOC analyst perspective and improving network security for my home.

I would probably go a lot further than a router, but in summer time, in my living space, having many devices to experiment with networking can quickly turn into a dry sauna and a suspicious electrical bill.

### Why would you do this?

Below are some practical benefits running OPNsense on your home network. It vastly improves safety for anyone at home using the internet.

 You have powerful control over all devices,

 Control who has access to the network, when and for how long,

 Limit insecure IOT devices access

 Filter traffic in and out of the network,

 Block adds for all devices,

 Block known malicious websites,

 Automatically scan the network for potentially malicious behavior,

 Receive prioritized alerts via email,

 And more...

A normal consumer router is no longer updated (patched for security) after 2 to 5 years. With OPNsense you have access to:

     Regular security updates for CVE's (common vulnerabilities and exposures),
     Features available in expensive commercial firewalls,
     Intrusion detection and prevention software like Suricata, Snort, Crowdsec, Zenarmour and many more.

Read more about it at [opnsense.org/about/about-opnsense](http://opnsense.org/about/about-opnsense)

The amount of features you can run is only limited by your choice of hardware.

### What about the downsides?

This is no walk in the park for someone with zero knowledge on IP addressing, subnets, firewall rules, DNS, network bridge and other networking notions. However I am also far from expert and learned a lot about networks along the way.

__This setup is not a install and forget type of deal. You have to regularly:__

- check for updates,
- check if any warnings popped up,
- sometimes a legitimate link might not work due to a false positive,
- ad-block lists can block legitimate links (I succeeded to block the mac os updates this way)

### A bit about the hardware.

Initially I wanted a mini PC for small footprint, low power and low noise (this thing sits near my desk). Unfortunately it is very difficult to find parts for trusted mini PC brands, especially for what I needed: additional 4 port RJ45 network connections.

Some mods I found were also very difficult to install.

Fun fact: some popular mini PC brands from china were shipped with malware preloaded from factory. Found this randomly when researching mini PC network upgrades.

### Hardware requirements

Mostly based on researching how to comfortably run intrusion detection and prevention software:

- Multi core CPU, low powered
- Have lots of storage for logs, min 250 Gb
- Minimum 8 gig RAM
- Have at least 4 nice RJ45 ethernet ports
- Run cool and quiet

For the small budget assigned to this project, new hardware was out of the question, but found a good fit on the open market:

A Dell Optiplex 5050 with 16Gb RAM, some 250GB ssd. Already had installed a 4 port network card, a nice i5-7400T CPU. Can easily be extended with RAM, more network cards or whatever else is compatible with the motherboard slots. The CPU type T is also a low power version perfectly suitable for running a PC 24-7.

It's a bit overkill for my home lab, but the advantage of this is that it stays cool and quiet.

Have you tried setting up your own firewall/router? Share your experiences and tips in the comments below.

Until next time, safe networking.
