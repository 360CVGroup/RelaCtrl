## 📌 Prompt Examples for RelaCtrl

This section provides example prompts for using the RelaCtrl model under different control modes.

### 🔧 Conditional Control Mode
For the standard conditional control model (**RelaCtrl_PixArt_Canny**), the demo image and corresponding prompt examples are shown below:
```
1.exampel1.png 
a large, well-maintained estate with a red brick driveway and a beautifully landscaped yard. The property is surrounded by a forest, giving it a serene and peaceful atmosphere. The house is situated in a neighborhood with other homes nearby, creating a sense of community. In the yard, there are several potted plants, adding to the lush greenery of the area. A bench is also present, providing a place for relaxation and enjoyment of the surroundings. The overall scene is picturesque and inviting, making it an ideal location for a family home.

2.exampel2.png
a white dress displayed on a mannequin, showcasing its elegant design. The dress is a short, white, and lacy dress, with a fitted waist and a full skirt. The mannequin is positioned in the center of the scene, showcasing the dress's style and fit. The dress appears to be a wedding dress, as it is white and has a classic, elegant appearance.

3.exampel3.png
a beautiful blue bird perched on a branch, surrounded by a lush green field. The bird is positioned in the center of the scene, with its wings spread wide, showcasing its vibrant blue feathers. The branch it is perched on is filled with pink flowers, adding a touch of color to the scene. The bird appears to be enjoying its time in the serene environment, surrounded by the natural beauty of the field and flowers.
```

### 🎨 Style Control Mode
For the style-guided control model (**RelaCtrl_PixArt_Canny_Style**), the demo image and prompt examples are provided below:
```
1.style1.png
gufeng_A tranquil mountain range with snow-capped peaks and their clear reflection in a calm lake, surrounded by trees, creates a stunning, serene landscape.

2.style2.png
oil_A man stands in a moonlit snowy field with a scythe, gazing at the moon, amidst trees, exuding mystery.

3.style3.png
paint_A vintage car drives down a dirt road, dusting up as it passes, center stage with two people observing on the sides, evoking nostalgia.

4.style4.png
3d_The painting is a close-up portrait of a bearded man with a mustache, focusing on his facial features in a blurred background.
```

Note: When using a style image for inference, you must prepend a **style_** annotation before the actual prompt.
Available style options include:

- gufeng

- 3d

- paint

- oil

