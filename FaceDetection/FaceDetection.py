

class FaceDetector:
    def detect_faces(self,model, image):
        results = model(image)
        faces = []
        gap = 0            #some gap between model output and face in the input image
        for res in results:
            box = res.boxes
            coord = box.xyxy.numpy()
            for lst in coord:
                faces.append(image[int(lst[1])-gap:int(lst[3])+gap,int(lst[0])-gap:int(lst[2])+gap])
        return faces
