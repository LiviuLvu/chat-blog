---
title: "Home lab vs Cloud provider: My quest for affordable, practical DevOps Kubernetes experience"
date: "2025-03-26"
summary: "Reasons why I stopped using Azure and decided to use my home lab to build a portfolio and gain practical knowledge."
# description: ""
tags: ["kubernetes", "azure", "mac mini", "cloud costs", "home lab", "virtual machine", "devops"]
author: "Liviu Iancu"
weight: 1
series: ["Devops"]
# categories: [""]
# aliases: [""]
# cover:
#   image: images/img.png
#   caption: "Description [text](https://link.somewhere/)"
---

### Reasons why I stopped using Azure and decided to use my home lab to build a portfolio and gain practical knowledge.

Well one reason mostly, is cost. Big risk of incurring 500 to 1000+ euros cost per month.

For reference, using the [Azure pricing calculator](https://azure.microsoft.com/en-us/pricing/calculator), Just one Kubernetes cluster with 4 virtual machines, running 15 days (lets say i don't have time every day to tinker with it) can cost 200+ euros! Not counting additional resources like subnet, load balancer, gateway, storage, possible ingress, egress data, monitoring, and other gadgets I'm forgetting right now.

For that sum, I can easily buy cheap hardware with 32Gb RAM, decent CPU, SSD storage, and run more than one cluster, non stop with any additional services I would want to experiment with.

### Goal

My goal is to learn how things connect to form a production infrastructure in the real world. CI-CD pipelines, load balancing, security, monitoring health checks, database backups, and have a good practical understanding of the cloud environment around a containerized web app. Basically DevOps and [Kubernetes](https://kubernetes.io/).

I went through the AZ-400 Admin course. Found it difficult to follow, because of the very limited practical sandbox training. A lot of theory favoring power-shell commands but I'm used to Linux. The sandboxes started very slow or not at all. On top of that after 2 months in, somehow i got banned from opening the sandbox exercises and noticed a lot of people were receiving the same error.

Just reading and clicking through the lessons slides is too much theory and won't persist in memory. I attempted a az-400 simulated exam after that and no surprise, did not went well.

### Cloud test run

For more practice I opened an Azure account with free tier for one month, created some virtual machines, did a bit of networking, experimented with local Azure CLI and a few small Bicep templates. It was fun and that free month went by faster than you can type ip addr to view your network interfaces.

I was very careful to delete all resources to not go over the 200$ limit for the free tier.

In life you easily get distracted. Its easy to forget to switch off and delete the resources every time you have a spare one hour in the evening after work to learn.

After free tier expired, I was too stressed about accidental costs, considering the plan is to build a complete prod setup with ci-cd pipeline, monitoring, health check, load balancing and probably, Kubernetes later on.

### My home lab

So as an alternative to cloud for now, the best thing is to use my existing home lab to see how far I can push it without needing additional hardware. I have OPNsense as a next gen router and a Mac mini M4 16gb RAM that should be enough to explore the simplest setup of clusters.

A home lab can be as complex as I can build it, no need to open it to the internet. I sleep well knowing i don't have to remember to clean all resources, wait for them to be deleted and check again if i missed something still running outside of a resource group.

Creating [virtual machines in UTM](https://mac.getutm.app) seems to work very well. Tested [Vagrant with UTM plugin](https://naveenrajm7.github.io/vagrant_utm/#getting-started). After 3 commands I had a Ubuntu VM with working ssh connection, generated from a template file.

Of course Kubernetes at home is overkill when a simple container app will run just fine with software like [Podman](https://podman.io/), [Orbstack](https://docs.orbstack.dev/quick-start) or [Docker](https://docs.docker.com/get-started/docker-overview/). My reasons for doing this are:

+ Proof of experience for career development

+ Trying out Kubernetes self healing and automated rollbacks on Home Assistant and other services running at home

+ The power of creating multiple VM's and containers with one command

### Next steps

Plenty to do next, but most important step is to generate 4 virtual machines that will provide base for a Kubernetes cluster, test how the Mini M4 16gb can handle multiple running virtual machines. So far it proved great for running Home Assistant in a VM.

Then will attempt to automate installing Kubeadm control-plane and nodes, try to run home assistant container version and follow how it behaves after connecting to a few devices.

That's it for now, hope this will be part of a long series of Kubernetes journey.

Get in touch if you know devops communities. Would also love to connect and share experiences with other people running a home lab with Kubernetes.
