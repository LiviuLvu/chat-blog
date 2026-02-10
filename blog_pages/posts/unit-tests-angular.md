---
title: "Unit tests examples in Angular with Jasmine"
date: "2019-12-05"
summary: "For you if the importance of unit testing is unquestionable and you are looking for unit testing references with examples of code."
# description: ""
tags: ["JavaScript", "Unit Testing", "Jasmine", "Angular", "Stackblitz"]
author: "Liviu Iancu"
weight: 5
series: ["Programming"]
# categories: [""]
# aliases: [""]
# cover:
#   image: images/img.png
#   caption: "Description [text](https://link.somewhere/)"
---
This article is dedicated to the misfits, the outliers, the… just kidding. It is for you if the importance of unit testing is unquestionable and you are looking for unit testing references with examples of code.

Below is a demo project that is using Jasmine for testing Angular. It is up and running, ready to tweak, experiment, break apart and let your imagination explore it.

It can be very useful to use as boilerplate in case you need to demo a concept related to your main project, or ask a question on stack overflow and provide live code to help test the solution.

https://stackblitz.com/edit/angular-unit-testing-examples

The above stackblitz project is also available on github. After download you can run the tests locally.

```
git clone https://github.com/LiviuLvu/angular-unit-testing-examples.git
npm run start
npm run test
```

### Best practice notes

Don’t test implementation details.

Recommended size is 10 lines of code or less.

Write expected functionality of a component first.

Reuse code as much as possible.

Description must be easy to read.

Code coverage recommended is minimum 80%.

Tests should run fast (especially on large projects).

Avoid complicated logic.

### Test structure recommended

Divide your test method into three sections:

```js {linenos=true}
// Arrange
let count = 0;// Act
const newCount = increment(count);// Assert
expect(newCount).toBe(1);
```

### Tests should not be fragile:

Lets say we test a line that creates some string.
```js {linenos=true}
let someString = `Active user is ${name}`
```
You can test only the dynamic content. In case the rest of the string is changed, it will not break test functionality.
```js {linenos=true}
expect(someString).toContain('Some Name');
```

### What’s in a good test failure bug report:

What were you testing?  
What should it do?  
What was the output (actual behaviour)?  
What was the expected output (expected behaviour)?  

### References and mentions  
__Articles__

An Overview of JavaScript Testing in 2019  
https://medium.com/welldone-software/an-overview-of-javascript-testing-in-2019-264e19514d0a

What every unit testing needs  
https://medium.com/javascript-scene/what-every-unit-test-needs-f6cd34d9836d

Angular unit testing 101 (with examples)  
https://dev.to/mustapha/angular-unit-testing-101-with-examples-6mc

__Video__

Kent C. Dodds — Write tests. Not too many. Mostly integration 
https://youtu.be/Fha2bVoC8SE?t=105

Unit Testing in JavaScript and Jasmine | TLDR Jasmine Unit Test Tutorial By: Dylan Israel  
https://www.youtube.com/watch?v=h2eWfvcAOTI

__Types of tests__

Functional Testing types include  
https://www.softwaretestinghelp.com/types-of-software-testing/  
Unit Testing, Integration Testing, System Testing, Sanity Testing, Smoke Testing, Interface Testing, Regression Testing, Beta/Acceptance Testing

Non-functional Testing types include  
https://www.softwaretestinghelp.com/types-of-software-testing/  
Performance Testing, Load Testing, Stress Testing, Volume Testing, Security Testing, Compatibility Testing, Install Testing, Recovery Testing, Reliability Testing, Usability Testing, Compliance Testing, Localization Testing

### Index

Fixture: A fixture is a wrapper around an instance of a component. With a fixture, we can have access to a component instance as well as its template.

Mock: Mock objects are fake (simulated) objects that mimic the behaviour of real objects in controlled ways.

Spy: Spies are useful for verifying the behaviour of our components depending on outside inputs, without having to define those outside inputs. They’re most useful when testing components that have services as a dependency.