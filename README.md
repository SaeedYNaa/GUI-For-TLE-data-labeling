## UI LABELING

This reprository contains the implementations of the annotation scheme for the Aramaic Incantation Bowls images introduce on our paper [Detecting Spiral Text Lines in Aramaic Incantation Bowls](https://link.springer.com/chapter/10.1007/978-3-031-78495-8_16). (Published at ICDAR24)


Our innovative annotation scheme based-UI Labeling designed to streamline and simplify the process of annotating bowl images with precision and efficiency. We understand the significance of accurate data in modern computer vision and machine learning tasks, and we are proud to present a user-friendly solution that label and segment polygons on bowl images.

### Here's how it works:
The UI Labeling scripts is called UIDataLabelingScript.py which resides in a folder (no matter what is it called) alongside the datasets.
![image8](https://github.com/SaeedYNaa/GUI-For-TLE-data-labeling/assets/45369975/3581f7e5-3bb4-4554-b9f3-c1c913dad46d)



#### Running the script:
Once the structure file is ready, open the terminal/cmd in the main directory then run the script by typing the following command:

```` python UIDataLabelingScript.py dataset_path/ saved_result_path/ ````

 where dataset_path is the path of the bowls dataset, and saved_result_path is the path of the saved result (the script will create such a folder if it doesn’t exist.)

Once running the script, the following canvas appears:

![image1](https://github.com/SaeedYNaa/GUI-For-TLE-data-labeling/assets/45369975/2c5d2626-c1c2-4c30-91de-410438a4ef96)
Now to annotate the data, we need to draw the polygons using the mouse operation and some predefined key-words.
#### Operations:
1. To draw a polygon on the canvas, use the mouse to get to the starting point you want, once you are there, left mouse button to draw an edge of the polygon, if there is more than one edge, the line between them is drawn. 
![image2](https://github.com/SaeedYNaa/GUI-For-TLE-data-labeling/assets/45369975/7e89d96c-ba04-4e9f-ba70-55178fa33d66)


2. Ctrl + z for undoing the last drawn edge:

   ![image3](https://github.com/SaeedYNaa/GUI-For-TLE-data-labeling/assets/45369975/12f430e6-35d3-421f-8a22-3770aebfb57c)


4. Zooming in functionality: In order to zoom the image, one should press ctrl + mouse wheel (up for zooming in, down for zooming out).

   ![image4](https://github.com/SaeedYNaa/GUI-For-TLE-data-labeling/assets/45369975/27972c3f-99c3-4f45-8224-783075d2c906)


5. Zooming + pan: once zooming, you can navigate the zoomed image using ctrl + right mouse button or you can use both scroll bars horizontally and vertically:

   ![image5](https://github.com/SaeedYNaa/GUI-For-TLE-data-labeling/assets/45369975/af282563-7980-4075-ac6c-150d788237dc)


6. Creating new polygon: you can create a new polygon using the key-word letter ‘n’ in order to create new polygon and to start from another point:
   
   ![image6](https://github.com/SaeedYNaa/GUI-For-TLE-data-labeling/assets/45369975/7be6ddbe-8c6a-422d-a596-90d11bb5d3fc)


7. Saving option: make sure you save the image after you finish annotating all the lines inside the image:
 ![image7](https://github.com/SaeedYNaa/GUI-For-TLE-data-labeling/assets/45369975/33161c99-ad71-4a6c-b58c-ced4414cc217)



## Citation
````
@InProceedings{nammneh2025spiraltextlines,
author="Nammneh, Said and Madi, Boraq and Atamni, Nour and Boardman, Shoshana and Vasyutinsky-Shapira, Daria and Rabaev, Irina and Saabni, Raid and El-Sana, Jihad",
title="Detecting Spiral Text Lines in Aramaic Incantation Bowls",
booktitle="Pattern Recognition",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="250--264",
isbn="978-3-031-78495-8"
}
````


