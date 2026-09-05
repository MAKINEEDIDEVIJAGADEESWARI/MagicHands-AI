import cv2
cap=cv2.VideoCapture(0)
while True:#Eventually we'll add a way to stop the loop when you press a key.
    ret , frame=cap.read()#ret->frame captured successfully,frame->the actual frame captured
    #"Show this camera frame in a window called MagicHands AI."
    cv2.imshow("MagicHands AI",frame)#cv2.imshow:-Display the frame Mgic Hands:-window name Frame:-the image received from the camera
    if cv2.waitKey(1) & 0xFF==ord('q'):#it stops the webcam when we click 'q' key or it continues to run the loop until we press 'q' key.
        break
cap.release()#release the camera
cv2.destroyAllwindows()#destroy all the windows opened by cv2.imshow
 