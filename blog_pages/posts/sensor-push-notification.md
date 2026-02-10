---
title: "Push notifications from sensors using python automation"
date: "2026-01-10"
summary: "Practical, fun way to use Python to solve problems in home automation."
# description: ""
tags: ["zigbee2mqtt", "mqtt", "python", "Push notification", "Proxmox", "LXC containers"]
author: "Liviu Iancu"
weight: 2
series: ["Programming"]
# categories: [""]
# aliases: [""]
# cover:
#   image: images/img.png
#   caption: "Description [text](https://link.somewhere/)"
---
  
**About this project**  
- Practical, fun way to use Python to solve problems in home automation.  
- Needed a reliable way to monitor and get alerts based on sensors i use around the house.   

**Why not Homeassistant?**  
- It's inconvenient to use dedicated device only for it Homeassistant.  
- Inside Proxmox container, it is difficult to pair sensors, displays duplicates, and can brake after updates.  

**Source files for the script and docker setup**  
[https://github.com/LiviuLvu/sensor-notifications](https://github.com/LiviuLvu/sensor-notifications)   

## Components needed for my Proxmox setup:  
  
### 1 Zigbee antenna USB  
Already have a usb from Conbee II which works great.  
Check its recognized and available in Proxmox:  
List all usb devices and search for your specific device:  
```
ls /dev/serial/by-id/ 
```
  
### 2 MQTT message broker  
What it is:  
central server for the Message Queuing Telemetry Transport protocol, lightweight publish-subscribe protocol designed for IOT devices limited in power and low bandwith.  

Selected option: **Mosquito**  
[https://mosquitto.org/](https://mosquitto.org/)  
  Proxmox LXC helper: [https://community-scripts.github.io/ProxmoxVE/scripts?id=mqtt](https://community-scripts.github.io/ProxmoxVE/scripts?id=mqtt)  
  Chose this one because it was available in Proxmox helpers and required less resource than the alternative EMQX.  
**Post install steps:**  
[https://github.com/community-scripts/ProxmoxVE/discussions/782](https://github.com/community-scripts/ProxmoxVE/discussions/782)   
  
### 3 MQTT client  
What it is:  
software that enables connection to a MQTT broker.  

Selected software: **zigbee2mqtt**  
[https://www.zigbee2mqtt.io/](https://www.zigbee2mqtt.io/)   
  Docker compose [https://www.zigbee2mqtt.io/guide/installation/02_docker.html](https://www.zigbee2mqtt.io/guide/installation/02_docker.html)  
  Proxmox LXC helper: [https://community-scripts.github.io/ProxmoxVE/scripts?id=zigbee2mqtt](https://community-scripts.github.io/ProxmoxVE/scripts?id=zigbee2mqtt)  
  This was the only zigbee - mqtt software available in proxmox helpers.  
**Post install steps:**  
[https://github.com/community-scripts/ProxmoxVE/discussions/410](https://github.com/community-scripts/ProxmoxVE/discussions/410)   
  
### 4 Push notification option  
Options:  
**Pushover:** Small multiplaform app, seems convenient  
[https://pushover.net/](https://pushover.net/)   
Telegram: Never used, not sure its worth creating account  
Email: Delayed, might miss something urgent like a pipe leak  
  
### Mosquitto lxc container  
Mapped in OPNsense on: 192.168.2.40:1883   
Details:  
```
Container Type: Unprivileged
Container ID: 101
Hostname: mqtt   

Resources:       
  Disk: 2 GB     
  CPU: 1 cores   
  RAM: 512 MiB   

Network:         
  Bridge: vmbr0  
  IPv4: 192.168.2.40/24 
  IPv6: none     

  Features:        
    FUSE: no | TUN: no    
  Nesting: Enabled | Keyctl: Disabled
  GPU: no | Protection: No  

  Advanced:        
    Timezone: Europe/Bucharest
  APT Cacher: no 
    Verbose: no    

```
  
### Zigbee2mqtt lxc container  
Mapped in OPNsense on: 192.168.2.42:9442   
Details:  
```
Container Type: Privileged
Container ID: 102
Hostname: zigbee2mqtt
Resources:
  Disk: 5 GB  
    CPU: 2 cores
    RAM: 1024 MiB  

Network:  
  Bridge: vmbr0  
  IPv4: 192.168.2.42/24 
  IPv6: none  
  Features: 
  FUSE: no | TUN: no 
    Nesting: Enabled | Keyctl: Disabled
  GPU: no | Protection: No

Advanced: 
  Timezone: Europe/Bucharest
  APT Cacher: no 
  Verbose: yes

```
  
### Debugging  
Check Logs: journalctl -u zigbee2mqtt -n 50 --no-pager  
Check config yaml settings:  
[https://www.zigbee2mqtt.io/guide/configuration/all-settings.html?utm_source=copilot.com#advanced](https://www.zigbee2mqtt.io/guide/configuration/all-settings.html?utm_source=copilot.com#advanced)  
  
**Zigbee2mqtt service issues installing and starting**  
 - specified static ip. It was not connecting to Opnsense DHCP  
 - accidentaly activated ssh, then had to deactivate it manually from inside container  
 - conbee II usb was used in another container. I had HomeAssistant running there  
 - frontend host set to internal "Localhost" instead of 0.0.0.0  
 - permit_join was set to false. Allows devices to join network  
 - reject_unauthorized was set to true, but there is not SSL cert setup yet  
 - wrong serial port. Did not put the full path  
  
To look out for:  
 - at least one sensor drains the battery too fast (4 days). zigbee2mqtt setup seems ok  
 - pushover api might reduce/drop free quota  
  
Changes made to the postinstall config example (bold text):  
```
version: 4
homeassistant:
  enabled: false
permit_join: true
frontend:
  enabled: true
  port: 9442
  host: 0.0.0.0
mqtt:
  base_topic: zigbee2mqtt
  server: mqtt://192.168.2.40:1883
  user: youruser
  password: yourpass
  keepalive: 60
  reject_unauthorized: false
  version: 4
serial:
  port: /dev/serial/by-id/usb-dresden_elektronik_ingenieurtechnik_GmbH_ConBee_II_DE2478649-if00
  adapter: deconz
advanced:
  pan_id: GENERATE
  network_key: GENERATE
  channel: 20

```
  
### Python libraries  
paho-mqtt docs:  
[https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html](https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html)  
requests docs:  
[https://requests.readthedocs.io/en/latest/](https://requests.readthedocs.io/en/latest/)   
  
**Successful logs inside container** from running python script:  
```
Connecting to MQTT...
Connected to MQTT with result code: Success
Message received: zigbee2mqtt/0xa4c13875a846a8f4 b'{"battery":68,"battery_low":false,"linkquality":255,"water_leak":false}'

```
  
**Checklist to validate minimal setup:**  
✓ Install client  
✓ install broker  
✓ pair sensor to zigbee2mqtt, check state  
✓ write script and test notification in dev container  
✓ containerize script in python based image  
✓ test script in container on macos  
✓ install push notification app, get api key  
✓ use script to push received message to phone  

Optional:  
. deploy script on Proxmox Dokploy container via ghrc  
. send email fallback  
. pair homeassistant to zigbee  
