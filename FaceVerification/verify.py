from Shared.utils import find_cosine_distance,find_euclidean_distance


class FaceVerification:

    def verify(self,emb1,emb2,metric='euclidean'):

        if metric=='euclidean':
            threshold = 0.9
            distance = find_euclidean_distance(emb1.numpy().reshape(-1,),emb2.numpy().reshape(-1,))
            if distance < threshold:
                return 1
            else:
                return 0
        
        elif metric=='cosine':
            threshold = 0.52
            distance = find_cosine_distance(emb1.numpy().reshape(-1,),emb2.numpy().reshape(-1,))
            if distance < threshold:
                return 1
            else:
                return 0
        
        