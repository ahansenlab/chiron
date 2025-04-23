import generate_gt as gt
import matplotlib.pyplot as plt
import centering_archival
import numpy as np

# Example usage: Generate 10 Gaussian-like blobs
size = (1, 1000, 1000)  # Size of the 3D volume
num_blobs = 500  # Number of blobs
sigma = 2 # Standard deviation of the blobs
ws = (7, 5, 5)  # Mask radius in z, y, and x. Radius gets added 1 so (7, 5, 5)  the mask is 15x11x11
# Step 1: Generate the ground truth set
image, ground_truth_centers = gt.generate_ground_truth_set(num_blobs, size, sigma)
#print(ground_truth_centers)
plt.figure()
plt.matshow(image[0,:,:])
plt.show()

imxy=image[0,:,:]

# Step 2: Generate initial integer guesses by rounding the ground truth centers
initial_guesses_centers = gt.initial_guesses(ground_truth_centers)
#print(initial_guesses_centers)
#initial_guesses_centers = [(c[0]+3.9, c[1]-4, c[2]+4) for c in initial_guesses_centers]
WINFRAC=15
ITER=5
iters=[1, 3, 5]

pos={}
diffs={}
for j in iters:
    pos[j] = []
    diffs[j] = []
plotting=False
for ind, i in enumerate(initial_guesses_centers):
    #print(ind)x
    if ind > 0 :
        xyc = (i[1], i[2]) # 1 is y, 2 is x
        im_sub = imxy[(xyc[0] - 19):(xyc[0] + 20), (xyc[1] - 19):(xyc[1] + 20)]
        if ind%100==0:
            # plt.matshow(im_sub)
            # plt.show()
            plotting=True
        for j in iters:

            cctr, shift, dist, fshifts = centering.centering_fracshift_track_distances_2(im_sub, win=WINFRAC, plotting=False,
                                                                              iterations=20, mask_radius=j)
            new_ctr = (xyc[0]+shift[0], xyc[1]+shift[1])
            print(f"Before Fracshift: {xyc}, After Fracshift: {new_ctr}, Shift = {shift}")
            # plt.figure()
            # plt.plot(np.arange(1, len(dist)+1, 1), dist)
            # plt.show()

            pos[j].append(new_ctr[0])
            pos[j].append(new_ctr[1])
            diff = np.linalg.norm(np.array(new_ctr) - np.array((ground_truth_centers[ind][1], ground_truth_centers[ind][2])))

            diffs[j].append(diff)
        print(f"True center: {ground_truth_centers[ind][1], ground_truth_centers[ind][2]}")


        #plt.figure()
        #plt.matshow(imxy[(xyc[0]-9):(xyc[0]+10), (xyc[1]-9):(xyc[1]+10)])
        #plt.scatter(9, 9)
        #plt.show()



for i in iters:
    poss = pos[i]
    posf, posi = np.modf(poss)
    plt.figure()
    plt.hist(posf, bins=100)
    plt.show()

plt.figure()
for i in iters:
    pcurr = diffs[i]

    plt.hist(pcurr, bins=50, alpha=0.4)
plt.legend(['r=1', 'r=3', 'r=5', 'r=7'])
plt.show()
