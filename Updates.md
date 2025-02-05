# Riley Herbst Masters Project
- This project will be a collaberation with the Field Museum and with Alex Wolpert and Matt von Konrat. 

- All Updates will be under the  Updates.md file.
****
- September 13th, Added Pre-Processing and simple OCR for bounding boxes. Under TextSegTesting.

- September 20th, Updates to TextSeg testing and Sam Model Setup. This is our image segmentation process.

- September 27th, Using LLM Whisperer And a structured extraction and parser using OpenAI. This process needs to be done in a Linux system so it resides in WSL. Code is available on here but if not on windows system it will NOT WORK.

- October 8th, Adding Automated Image Cropping based on segmented Images. (Look into colors for borders maybe some are different from others in effectiveness??)

- Oct 11th, SAM model is able to segment Images and stores each segment in a list. Taking the two/three largest segments and combining them together seems to be a good idea. BUT I need to specify the segmentation to just regions of text. Doing some test of segmentation seems to take from 17-39 Seconds Per Image. If this was hosted on some server would be a LOT faster.

- Oct 14th, Detectron introduction and finding out ease of finetuning. 

- Oct 20th, Draft of Thesis Proposal Started

- Nov 10th, Grouning Dino Testing with text prompts for SAM

- Nov 15th, Writing Thesis Proposal. Members Include Professor Schnepp, Professor Dantsin, Professor Hajizadeh. 

- Nov 25th, Propsal Submitted

- Dec 14th, EoS, on break

****
### Second Semester:

Jan 21st, Start Cross validation pipeline with three models, discuss AWS

Jan 30th, Layout algorithm for cross validation of 5 models. Claude, Gpt, Llama3.2V-ision, Amazon-Nova-Pro, 