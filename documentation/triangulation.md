## Triangulation

#### Purely co-ordinate based (Very basic start).

This involves predicting a co-ordinate on the screen using an eye tracking module. Based on the co-ordinates take an action.
This is almost like using a mouse and heavily depends on the accuracy of the eye tracker.
As eye tracker Hence might not be very practical.

###### Note
From here on an idea of what the available objects of interests on the screen are to be known to the software. This can be done by
1. the underlying software broadcasting the objects of interest (Like a webpage could broadcast all the buttons). 
2. A middleware -  layer of the Operating system could broadcast the OI co-ordinates for all the buttons / chrome extention could to that for all the links. 
3. Also if none of this is possible the Pixel values of the screen can be captured and a passed through a neural network to identify the object of interest. Also the search region for this could be reduced by taking into account a rough idea of where the user is looking
####  Nearest OI

Use the above method to detect a co-ordinate, find the most nearest Object of Interest and highlight it. So the user knows the OI which is selected and the user can take action.

#### Classifier

Use a classifier to detect the OI which the user is looking at. The classifier could be a soft max over all the OIs present.