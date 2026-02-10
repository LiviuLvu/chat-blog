---
title: "Git pull-request from command line"
date: "2018-05-19"
summary: "A more direct pr request init from terminal"
# description: ""
tags: ["Git", "Terminal", "Version Control"]
author: "Liviu Iancu"
weight: 5
series: ["Programming"]
# categories: [""]
# aliases: [""]
# cover:
#   image: images/img.png
#   caption: "Description [text](https://link.somewhere/)"
---
One of the projects I was working on has the default pull-request option set to master.
For whatever reason I kept forgetting to set this to develop branch. It just felt like it would stay set to develop once i created a PR with that setting…silly me.

Getting frustrated with my forgetfulness i searched for a more convenient, alternative way to send pull requests from the command line. This is how I found

```
git pull-request
```
<em>https://github.com/jd/git-pull-request  
‘git-pull-request is a command line tool to send GitHub pull-request from your terminal’</em>

Naturally like any experiment i try, it is too good to be true when it works just by copy-pasting the commands from the instructions, because i don’t understand a lot of stuff in those instructions.

I got some `missing .netrc` file error. Bummer. Now I told myself: OK, if it takes me more than 10 minutes to get this pull-request from the command line working I will skip this experiment and move ahead with more important stuff on my ‘To learn list’ (darn list, it’s starting to look like a sky scraper).

16 minutes later … multiple tabs open, git settings files, .gitconfig, .netrc, security issues, more confusion … time out.

Beyond a point I get to involved in the activity, or rather too attached to the idea to give up. Especially if it would increase productivity and remove manual input. In this case, after a break I returned to the screen and found clarity.

### Setup shenanigans
Below I will detail some of the steps to setup that worked for me on mac
Tried to follow the instructions on: https://github.com/jd/git-pull-request

Installation — this went like a charm: `pip3 install git-pull-request`
* But it worked for me because I already had Python3 installed during a weekend Hackaton :) that also took some focus to set up correctly.

This command line tool needs a config file `.netrc` placed in user path `~/`
It looks like this:

```
machine github.com
editor nano
login <git user name>
password <personal access token>machine api.github.com
editor nano
login <git-user-name>
password <personal access token>
```

I tried it without editor and it kept throwing some error:

```
$EDITOR is unset, you will not be able to edit the pull-request message
```

So i thrown in that setting to make it go away.

The personal access token can be your plain text password but it is not recommended.
Instead you can generate a token in git > developer settings > Personal access tokens > Generate new token
Some also recommended to encrypt the folder where you keep these settings

After you have .netrc file in place you should be able to use the `git pull-request` command.

I used it like this:
*switch to local branch with commits pushed to remote branch, ready for PR

```
git checkout <local-branch-tracking-upstream-branch>
git pull-request — target-branch develop — no-rebase
```
### Git pull-request options
```
usage: git-pull-request [-h] [ — download DOWNLOAD] [ — debug] 
 [ — target-remote TARGET_REMOTE] 
 [ — target-branch TARGET_BRANCH] [ — title TITLE] 
 [ — message MESSAGE] [ — no-rebase] [ — force-editor] 
 [ — no-comment-on-update] [ — comment COMMENT] 
 [ — no-tag-previous-revision] 
 
Send GitHub pull-request. 
 
optional arguments: 
 -h, — help show this help message and exit 
 — download DOWNLOAD, -d DOWNLOAD 
 Checkout a pull request 
 — debug Enabled debugging 
 — target-remote TARGET_REMOTE 
 Remote to send a pull-request to. Default is auto- 
 detected from .git/config. 
 — target-branch TARGET_BRANCH 
 Branch to send a pull-request to. Default is auto- 
 detected from .git/config. 
 — title TITLE Title of the pull request. 
 — message MESSAGE, -m MESSAGE 
 Message of the pull request. 
 — no-rebase, -R Don’t rebase branch before pushing. 
 — force-editor Force editor to run to edit pull-request message. 
 — no-comment-on-update 
 Do not post a comment stating the pull-request has 
 been updated. 
 — comment COMMENT, -C COMMENT 
 Comment to publish when updating the pull-request 
 — no-tag-previous-revision 
 Preserve older revision when pushing 
```
Also tried
```
git pull-request — target-branch master — no-rebase — force-editor
```
But this throws error:
```
bad follower token ‘editor’ (/Users/my.user/.netrc, line 2)
```
I wrote the people from https://github.com/jd/git-pull-request
If you know how to fix this please send a message.

That’s all for this experiment.
Hope this helps you to be more productive. Have fun!

Update:

To fix the editor is unset error, add this line: ```export EDITOR=nano```

inside this file: ~/.bash_profile (mac OS)

and now whenever I write the PR command, the nano editor opens and… behold! I can edit the PR request message.