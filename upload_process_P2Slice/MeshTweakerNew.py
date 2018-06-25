## Author: Christoph Schranz, Salzburg Research, 2015/16
## Contact: christoph.schranz@salzburgresearch.at
## Runs on: Python 2.7 and 3.5

#import sys
import math
from time import time, sleep
import re
from collections import Counter
# upgrade numpy with: "pip install numpy --upgrade"
import numpy as np
import json
import logging
#import time
from itertools import compress

from slicing_config import TWEAKER_OVERHANG_ANGLE_DEGREE

# Constants used:
VECTOR_TOL = 0.001  # To remove alignment duplicates, the vector tolerance is
                    # used to distinguish two vectors.
PLAFOND_ADV = 0.0   # Printing a plafond is known to be more effective than
                    # very step overhangs. This value sets the advantage in %.
first_lay_h = 0.2   # Since the initial layer of a print has a higher altitude
                    # >= 0, bottom layer and very bottom-near overhangs can be
                    # handled as similar.
NEGL_FACE_SIZE = 1  # The fast operation mode neglects facet sizes smaller than
                    # this value (in mm^2) for a better performance
ABSOLUTE_F = 100    # These values scale the the parameters bottom size,
RELATIVE_F = 1      # overhang size, and bottom lenght to get a robust


class Tweak:
    """ The Tweaker is an auto rotate class for 3D objects.

    The critical angle CA is a variable that can be set by the operator as
    it may depend on multiple factors such as material used, printing
     temperature, printing speed, etc.

    Following attributes of the class are supported:
    The tweaked z-axis'.
    Euler coords .v and .phi, where v is orthogonal to both z and z' and phi
     the angle between z and z' in rad.
    The rotational matrix .Matrix, the new mesh is created by multiplying each
     vector with R.
    And the relative unprintability of the tweaked object. If this value is
     greater than 10, a support structure is suggested.
        """
    def __init__(self, content, extended_mode=False, verbose=True,
                 favside=None, logging=logging):

        self.extended_mode = extended_mode
        n = -np.array([0,0,1], dtype=np.float64)
        orientations = [[list(n), 0.0]]

        ## Preprocess the input mesh format.
        t_start = time()
        mesh = self.preprocess(content)
        if favside:
            mesh = self.favour_side(mesh, favside)
        t_pre = time()


        ## Searching promising orientations:
        orientations += self.area_cumulation(mesh, 10)

        t_areacum = time()
        if extended_mode:
            dialg_time = time()
            orientations += self.death_star(mesh, 8)
            orientations += self.add_supplements()
            dialg_time = time() - dialg_time

        t_ds = time()

        orientations = self.remove_duplicates(orientations)

        logging.debug("Examine {} orientations:".format(len(orientations)))
        logging.debug("  %-26s %-10s%-10s%-10s " %("Alignment:",
            "Bottom:", "Overhang:", "Unpr.:"))

        datatweaker = json.loads(json.dumps({}))
        i = 0
    
        # Calculate the unprintability for each orientation
        results = np.array([None,None,None,np.inf])
        for side in orientations:
            start = time()
            orientation =np.array([float("{:6f}".format(-i)) for i in side[0]])
            mesh = self.project_verteces(mesh, orientation)
            bottom, overhang = self.lithograph(mesh, orientation)
            Unprintability = self.target_function(bottom, overhang)
            print("Unpr. : ",Unprintability, " Overhang : ", overhang, "Bottom : ", bottom, "Orientation : ", orientation.tolist())
            results = np.vstack((results, [orientation, bottom,
                            overhang, Unprintability]))
            logging.debug("  %-26s %-10s%-10s%-10s "
            %(str(np.around(orientation, decimals = 4)),
            round(bottom, 3), round(overhang,3),
            round(Unprintability,2)))

            i = i +1
            datatweaker.update({'orientation_{}'.format(i) : {'orientation' : orientation.tolist(), 'bottom_area' : bottom, 'overhang' : overhang, 'unprintability' : Unprintability}})
            end = time()
            #print("For 1 orientation : ", end-start)
        t_lit = time()
        
        # Best alignment
        # best_alignment = self.postprocess_results(results, 1)
        if np.isinf(np.min(results[:, 3])):
            raise ValueError("all Unprintability is inf, cannot find a good orientation")
        else:
            best_alignment = results[np.argmin(results[:, 3])] 
        logging.debug(" Time-stats of algorithm: Preprocessing:{pre:2f} s Area Cumulation:{ac:2f} s Death Star:{ds:2f} s Lithography Time: {lt:2f} s Total Time: {tot:2f} s ".format(pre=t_pre-t_start, ac=t_areacum-t_pre,
           ds=t_ds-t_areacum, lt=t_lit-t_ds, tot=t_lit-t_start))

        #Checking if there are two or more close unprintability
        close_unprintability_orientations = []
        for alignment in results:
            if (alignment[3]-best_alignment[3])/best_alignment[3] < 1:
                close_unprintability_orientations.append(alignment)
        if len(close_unprintability_orientations) > 1:
            print("Computing the volumes")
            vol = self.computeSupportVolume(mesh, best_alignment[0])
            unprMin = self.target_function(best_alignment[1], vol)
            for alignment in close_unprintability_orientations:
                vol = self.computeSupportVolume(mesh, alignment[0])
                unpr = self.target_function(alignment[1], vol)
                print("Unpr. : ", unpr, "Volume : ", vol, "Bottom : ", alignment[1], "Orientation : ", alignment[0].tolist())
                if unpr < unprMin:
                    unprMin = unpr
                    best_alignment = alignment

        if len(best_alignment) > 0:
            [v, phi, Matrix] = self.euler(best_alignment)
            self.Euler = [[v[0],v[1],v[2]], phi]
            self.Matrix = Matrix

            self.Alignment=best_alignment[0]
            self.BottomArea = best_alignment[1]
            self.Overhang = best_alignment[2]
            self.Unprintability = best_alignment[3]
            self.datatweaker = json.dumps(datatweaker)

        return None

    def postprocess_results(self, results, threshold):
        '''if Unprintability is within a threshold favour the one with bigger bottom area'''
        smallest_unprintability = np.min(results[:, 3])
        results_unpr_in_thredhold = results[results[:,3] < smallest_unprintability + threshold]
        # if original orientation is in results_unpr_in_thredhold use it, usually it is the first result
        if np.array_equal(results_unpr_in_thredhold[0][0], [0, 0, 1]):
            original_bottom_area = results_unpr_in_thredhold[0][1]
            max_bottom_area = np.max(results_unpr_in_thredhold[:, 1])
            # if bottom area is relatively similar then favour original orientation
            if abs(original_bottom_area - max_bottom_area) < 10:
                return results_unpr_in_thredhold[0]

        return results_unpr_in_thredhold[np.argmax(results_unpr_in_thredhold[:, 1])]


    def target_function(self, bottom, overhang):
        '''This function returns the Unprintability for a given set of bottom
        overhang area and bottom lenght, based on an ordinal scale.'''
        # bottom zero not blow up to zero
        #  Unprintability = overhang * 20/np.pi * (np.arctan(6.5 * (- bottom + 1/8)) + np.pi/2)
        #bottom = np.clip(bottom, 0, 100)
        #overhang = np.clip(overhang, 0, 100)
        if bottom == 0:
            Unprintability = np.inf
        else:
            Unprintability =  overhang * 0.01 / bottom

        return np.around(Unprintability, 6)


    def preprocess(self, content):
        '''The Mesh format gets preprocessed for a better performance.'''
        mesh = np.array(content, dtype=np.float64)

        # prefix area vector, if not already done (e.g. in STL format)
        if len(mesh[0]) == 3:
            row_number = int(len(content)/3)
            mesh = mesh.reshape(row_number,3,3)
            v0=mesh[:,0,:]
            v1=mesh[:,1,:]
            v2=mesh[:,2,:]
            normals = np.cross( np.subtract(v1,v0), np.subtract(v2,v0)
                                                    ).reshape(row_number,1,3)
            mesh = np.hstack((normals,mesh))

        face_count = len(mesh)

        # append columns with a_min, area_size
        addendum = np.zeros((face_count, 4, 3))

        # x
        addendum[:,0,0] = mesh[:,1,0]
        addendum[:,0,1] = mesh[:,2,0]
        addendum[:,0,2] = mesh[:,3,0]

        # y
        addendum[:,1,0] = mesh[:,1,1]
        addendum[:,1,1] = mesh[:,2,1]
        addendum[:,1,2] = mesh[:,3,1]

        # z
        addendum[:,2,0] = mesh[:,1,2]
        addendum[:,2,1] = mesh[:,2,2]
        addendum[:,2,2] = mesh[:,3,2]

        # calc area size
        addendum[:,3,0] = (np.sum(np.abs(mesh[:,0,:])**2, axis=-1)**0.5).reshape(face_count)
        addendum[:,3,1] = np.max(mesh[:,1:4,2], axis=1)
        addendum[:,3,2] = np.median(mesh[:,1:4,2], axis=1)
        mesh = np.hstack((mesh, addendum))

        # filter faces without area
        mesh = mesh[mesh[:,7,0]!=0]
        face_count = len(mesh)

        # normalise area vector and correct area size
        mesh[:,0,:] = mesh[:,0,:]/mesh[:,7,0].reshape(face_count, 1)
        mesh[:,7,0] = mesh[:,7,0]/2

        """ mesh is now a array with all the data of the stl files
        mesh[ ... [ [nx, ny, nz] normals from the file
                    [v1x, v1y, v1z] vertex 1 from the file
                    [v2x, v2y, v2z] vertex 2 from the file
                    [v3x, v3y, v3z] vertex 3 from the file
                    [v1x, v2x, v3x] x coordinates of the 3 vertices, projected
                    [v1y, v2y, v3y] y coordinates of the 3 vertices, projected
                    [v1z, v2z, v3z] z coordinates of the 3 vertices, projected
                    [face area, max(viz), med(viz)] depend on the projected vertices
                    ] ...
        ]]
        """

        if not self.extended_mode: # TODO remove facets smaller than a
                                #relative proportion of the total dimension
            filtered_mesh = mesh[mesh[:,7,0] > NEGL_FACE_SIZE]
            if len(filtered_mesh) > 100:
                mesh = filtered_mesh

        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return mesh

    def favour_side(self, mesh, favside):
        raise NotImplementedError
        '''This function weights one side, orientation closer than 45 deg
        are sized up.'''
        if type(favside)==type("str"):
            restring = r"(-?\d*\.{0,1}\d+)[, []]*(-?\d*\.{0,1}\d+)[, []]*(-?\d*\.{0,1}\d+)\D*(-?\d*\.{0,1}\d+)"
            x = float(re.search(restring, favside).group(1))
            y = float(re.search(restring, favside).group(2))
            z = float(re.search(restring, favside).group(3))
            f = float(re.search(restring, favside).group(4))

        norm = np.sqrt(np.sum(np.array([x, y, z])**2))
        side = np.array([x,y,z]/norm)
        logging.debug("You favour the side {} with a factor of {}".format(
            side, f))
        mesh[:,7,0] = np.where(np.sum(
            np.subtract(mesh[:,0,:], side)**2, axis=1) < 0.7654,
            f * mesh[:,7,0], mesh[:,7,0])
        return mesh


    def area_cumulation(self, mesh, n = None):
        '''Gathering the most auspicious alignments by cumulating the
        magnitude of parallel area vectors.'''
        if n is not None:
            if self.extended_mode: best_n = 10
            else: best_n = 7
        orient = Counter()

        align = mesh[:,0,:]
        for index in range(len(mesh)):       # Cumulate areavectors
            orient[tuple(align[index])] += mesh[index, 7, 0]
        top_n = orient.most_common(n)
        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return [[list(el[0]), float("{:2f}".format(el[1]))] for el in top_n]


    def death_star(self, mesh, best_n):
        '''Searching normals or random edges with one vertice'''
        vcount = len(mesh)
        # Small files need more calculations
        if vcount < 1000: it = 30
        elif vcount < 2000: it = 15
        elif vcount < 5000: it = 5
        elif vcount < 10000: it = 3
        elif vcount < 20000: it = 2
        else: it = 1

        vertexes = mesh[:vcount,1:4,:]
        v0u1 = vertexes[:,np.random.choice(3, 2, replace=False)]
        v0 = v0u1[:,0,:]
        v1 = v0u1[:,1,:]
        v2 = vertexes[:,np.random.choice(3,1, replace=False)].reshape(vcount,3)

        lst = list()
        for i in range(it):
            v2 = v2[np.random.choice(vcount, vcount),:]
            normals = np.cross( np.subtract(v2,v0), np.subtract(v1,v0))

            # normalise area vector
            area_size = (np.sum(np.abs(normals)**2, axis=-1)**0.5).reshape(vcount,1)
            nset = np.hstack((normals, area_size))

            nset = np.array([n for n in nset if n[3]!=0])
            if nset.size == 0:
                continue

            normals = np.around(nset[:,0:3]/nset[:,3].reshape(len(nset),1),
                                decimals=6)

            lst += [tuple(face) for face in normals]
            sleep(0)  # Yield, so other threads get a bit of breathing space.

        orient = Counter(lst)
        top_n = orient.most_common(best_n)
        top_n = list(filter(lambda x: x[1] > 2, top_n))

        # add antiparallel orientations
        top_n = [[list(v[0]),v[1]] for v in top_n]
        top_n += [[list((-v[0][0], -v[0][1], -v[0][2] )), v[1]]
                        for v in top_n]
        return top_n


    def add_supplements(self):
        '''Supplement 18 additional vectors'''
        v = [[0,0,-1], [0.70710678,0,-0.70710678],[0,0.70710678,-0.70710678],
             [-0.70710678,0,-0.70710678],[0,-0.70710678,-0.70710678],
    [1,0,0],[0.70710678,0.70710678,0],[0,1,0],[-0.70710678,0.70710678,0],
    [-1,0,0],[-0.70710678,-0.70710678,0],[0,-1,0],[0.70710678,-0.70710678,0],
            [0.70710678,0,0.70710678],[0,0.70710678,0.70710678],
             [-0.70710678,0,0.70710678],[0,-0.70710678,0.70710678],[0,0,1]]
        v = [[[float(j) for j in i],0] for i in v]
        return v


    def remove_duplicates(self, o):
        '''Removing duplicates in orientation'''
        orientations = list()
        for i in o:
            duplicate = None
            for j in orientations:
                # redundant vectors have an angle smaller than
                # alpha = arcsin(atol). atol=0.087 -> alpha = 5
                if np.allclose(i[0],j[0], atol = 0.087):
                    duplicate = True
                    break
            if duplicate is None:
                orientations.append(i)
        return orientations


    def project_verteces(self, mesh, orientation):
        '''Returning the "lowest" point vector regarding a vector n for
        each vertex.'''
        z = np.inner(orientation, np.array([0,0,1]))
        # create an oriented base
        if z == 1.0 :
            bx = np.array([1,0,0])
        elif z == -1.0 :
            bx = np.array([-1,0,0])
        else :
            bx = np.cross(orientation, np.array([0,0,1]))
            bx = bx / np.sum(np.abs(bx)**2)**0.5
        by = np.cross(orientation, bx)

        #Project of the vertices from the column 1 to 3 in column 4 to 6

        mesh[:,4,0] = np.inner(mesh[:,1,:], bx) # x0
        mesh[:,4,1] = np.inner(mesh[:,2,:], bx) # x1
        mesh[:,4,2] = np.inner(mesh[:,3,:], bx) # x2

        mesh[:,5,0] = np.inner(mesh[:,1,:], by) # y0
        mesh[:,5,1] = np.inner(mesh[:,2,:], by) # y1
        mesh[:,5,2] = np.inner(mesh[:,3,:], by) # y2

        mesh[:,6,0] = np.inner(mesh[:,1,:], orientation) # z0
        mesh[:,6,1] = np.inner(mesh[:,2,:], orientation) # z1
        mesh[:,6,2] = np.inner(mesh[:,3,:], orientation) # z2

        mesh[:,7,1] = np.max(mesh[:,6,:], axis=1)
        mesh[:,7,2] = np.median(mesh[:,6,:], axis=1)
        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return mesh







    def lithograph(self, mesh, orientation):
        

        """
        def candidateTriangle(point, triangle):
            if(np.sign(triangle[1][0]-point[0])==np.sign(triangle[2][0]-point[0])==np.sign(triangle[3][0]-point[0])):
                return False 
            if(np.sign(triangle[1][1]-point[1])==np.sign(triangle[2][1]-point[1])==np.sign(triangle[3][1]-point[1])):
                return False
            #if(np.sign(triangle[1][2]-point[2])==np.sign(triangle[2][2]-point[2])==np.sign(triangle[3][2]-point[2])):
            #    return False
            return True
            
            
        def findHeight(point, mesh, zmin):
                #point is as 3d array
            heightMin = np.inf
            rayVector = np.array([0, 0, -1])
            for triangle in mesh:
                if(point[2]-triangle[1][2]<heightMin or point[2]-triangle[2][2]<heightMin or point[2]-triangle[3][2]<heightMin) :
                    triangle = [triangle[1], triangle[2], triangle[3]]
                    t = rayIntersectsTriangle(point, rayVector, triangle)
                    if t>0 and t<heightMin:
                        heightMin = t
            if np.isinf(heightMin): 
                heightMin = point[2] - zmin
            return heightMin"""
           


        '''Calculating bottom and overhang area for a mesh regarding
        the vector n.'''
        overhang = 0
        bottom = 0

        xmax = np.amax(mesh[:,4,:])
        xmin = np.amin(mesh[:,4,:])
        ymax = np.amax(mesh[:,5,:])
        ymin = np.amin(mesh[:,5,:])
        zmax = np.amax(mesh[:,7,2]) # z median
        zmin = np.amin(mesh[:,6,:])

        # reference area
        # area = (xmax - xmin) * (ymax - ymin)
        total_area = np.sum(mesh[:,7,0])
        area = total_area

        # reference volume
        volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)

        # angle of support
        ascent = np.cos((180 - (90 - (TWEAKER_OVERHANG_ANGLE_DEGREE - 0.1)))*np.pi/180)
        anti_orient = -np.array(orientation)

        height = np.abs(zmax - zmin)
        first_lay_h = height/1000
        second_lay_h = height/100
        third_lay_h = height/50

        # filter bottom
        bottoms3 = mesh[mesh[:,7,1] < zmin + third_lay_h]
        bottoms2 = bottoms3[bottoms3[:,7,1] < zmin + second_lay_h]
        bottoms1 = bottoms2[bottoms2[:,7,1] < zmin + first_lay_h]
        if len(bottoms3) > 0:
            bottom = np.sum(bottoms1[:,7,0] * -np.inner(bottoms1[:,0,:], orientation)) * 0.25
            bottom += np.sum(bottoms2[:,7,0] * -np.inner(bottoms2[:,0,:], orientation)) *0.7
            bottom += np.sum(bottoms3[:,7,0] * -np.inner(bottoms3[:,0,:], orientation)) *0.05
            bottom = max(0,bottom) / area * 100
        else:
            bottom = 0

        """
        bottom area is commute on 3 différents layers:
        ^ height of the object
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |____________________________________________________ third_lay_h: 2% of the height, coefficient: 0.05
        |
        |____________________________________________________ second_lay_h: 1% of the height, coefficient: 0.75
        |
        |_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_- first_lay_h: 0.1% of the height, coefficient: 1
                                                            z min of the object projected
        """
        
        # filter overhangs
        overhangs = mesh[np.inner(mesh[:,0,:], orientation) < ascent]
        overhangs = overhangs[overhangs[:,7,1] > (zmin + first_lay_h)]
        
        #compute overhang with surface
        if len(overhangs) > 0:
            overhang = np.sum(overhangs[:, 7, 0])
            overhang = overhang/total_area * 100
        else: overhang = 0

        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return bottom, overhang

    def computeSupportVolume(self, mesh, orientation):
        
        def rayIntersectsTriangle(rayOrigin, rayVector, triangle):
            #rayOrigin : 3d numpy array
            #rayVector : 3d numpy array, normed
            #triangle = array of 3 vertex, where vertex are 3d numpy array
            #we return t, the distance between the point and the triangle
            #if there is no intersection, we return -1 
            EPSILON = 0.000000000000001
            vertex0 = np.array(triangle[0])
            vertex1 = np.array(triangle[1])
            vertex2 = np.array(triangle[2])
            edge1 = vertex1-vertex0
            edge2 = vertex2-vertex0
            h = np.cross(rayVector, edge2)
            a = np.inner(edge1, h)
            if a>-EPSILON and a<EPSILON:
                return -1
            f = 1/a
            s = rayOrigin - vertex0
            u = f * np.inner(s, h)
            if u<0 or u>1:
                return -1
            q = np.cross(s, edge1)
            v = f * np.inner(rayVector, q)
            if v<0 or u+v>1:
                return -1
            #We compute t
            t = f * np.inner(edge2, q)
            if t>EPSILON:
                #outIntersectionPoint = rayOrigin + rayVector * t
                return t 
            #t is the distance between the origin point and the hit point
            else:
                return -1

        #compute overhang with volume
        ascent = np.cos((180 - (90 - (TWEAKER_OVERHANG_ANGLE_DEGREE - 0.1)))*np.pi/180)
        zmin = np.amin(mesh[:,6,:])
        meshC = mesh[np.argsort(mesh[:,7,2])]
        L1 = np.inner(meshC[:,0,:], orientation) < ascent
        L2 = meshC[:,7,1] > (zmin + first_lay_h)
        L = L1*L2
        overhangs = meshC[L]
        numberOverhangs = len(overhangs)
        indexOverhangs = np.array(list(compress(range(len(L)), L)))
        volumeSupport = 0

        xmins = np.min(meshC[:,4,:], axis=1)
        xmaxs = np.max(meshC[:,4,:], axis=1)
        ymins = np.min(meshC[:,5,:], axis=1)
        ymaxs = np.max(meshC[:,5,:], axis=1)

        for i, overhang in enumerate(overhangs):
            indexOverhang = indexOverhangs[i]
            center = np.array(overhang[4:7,:]).sum(axis=1)/3

            """
            start_0 = time()
            candidateMesh = meshC[:indexOverhang]
            L1 = np.sign(candidateMesh[:,4,0]-center[0])!=np.sign(candidateMesh[:,4,1]-center[0])
            L2 = np.sign(candidateMesh[:,4,0]-center[0])!=np.sign(candidateMesh[:,4,2]-center[0])
            L3 = np.sign(candidateMesh[:,4,1]-center[0])!=np.sign(candidateMesh[:,4,2]-center[0])
            L = L1+L2+L3
            L1 = np.sign(candidateMesh[:,5,0]-center[1])!=np.sign(candidateMesh[:,5,1]-center[1])
            L2 = np.sign(candidateMesh[:,5,0]-center[1])!=np.sign(candidateMesh[:,5,2]-center[1])
            L3 = np.sign(candidateMesh[:,5,1]-center[1])!=np.sign(candidateMesh[:,5,2]-center[1])
            M = L1+L2+L3
            candidateMesh = list(candidateMesh[L*M])

            end_0 = time()
            """

            #  print("bool", end_0 - start_0)
            #  print("old num", len(candidateMesh))
            #  print(candidateMesh)

            #  start_new_bool = time()
            bool_x = np.logical_and(
                xmins[:indexOverhang] < center[0], xmaxs[:indexOverhang] > center[0]
            )
            bool_y = np.logical_and(
                ymins[:indexOverhang] < center[1], ymaxs[:indexOverhang] > center[1]
            )
            bool_xyz = np.logical_and(bool_x, bool_y)
            #  end_new_bool = time()
            candidateMesh = meshC[:indexOverhang][bool_xyz].tolist()

            #  print("new_bool", end_new_bool - start_new_bool)

            #  print("new", bool_xyz)
            #  print("new num", np.sum(bool_xyz))
            #  print(meshC[:indexOverhang][bool_xyz])

            #  start_1 = time()
            height = -1
            while len(candidateMesh) > 0 and height < 0:
                closerTriangle = np.array(candidateMesh[-1])
                closerTriangle = [closerTriangle[4:7,0], closerTriangle[4:7,1], closerTriangle[4:7,2]]
                t = rayIntersectsTriangle(center, np.array([0,0,-1]), closerTriangle)
                if t>0 :
                    height = t
                candidateMesh.pop()

            #  end_1 = time()

            #  print("time intersect", end_1 - start_1)

            if len(candidateMesh)==0 :
                height = center[2] - zmin

            vS = height * overhang[7, 0] 
            volumeSupport += vS
        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return volumeSupport

    def euler(self, bestside):
        '''Calculating euler rotation parameters and rotation matrix'''
        if np.allclose(bestside[0], np.array([0, 0, -1]), atol = VECTOR_TOL):
            v = [1, 0, 0]
            phi = np.pi
        elif np.allclose(bestside[0], np.array([0, 0, 1]), atol = VECTOR_TOL):
            v = [1, 0, 0]
            phi = 0
        else:
            phi = float("{:2f}".format(np.pi - np.arccos( -bestside[0][2] )))
            v = [-bestside[0][1] , bestside[0][0], 0]
            v = [i / np.sum(np.abs(v)**2, axis=-1)**0.5 for i in v]
            v = np.array([float("{:2f}".format(i)) for i in v])

        R = [[v[0] * v[0] * (1 - math.cos(phi)) + math.cos(phi),
              v[0] * v[1] * (1 - math.cos(phi)) - v[2] * math.sin(phi),
              v[0] * v[2] * (1 - math.cos(phi)) + v[1] * math.sin(phi)],
             [v[1] * v[0] * (1 - math.cos(phi)) + v[2] * math.sin(phi),
              v[1] * v[1] * (1 - math.cos(phi)) + math.cos(phi),
              v[1] * v[2] * (1 - math.cos(phi)) - v[0] * math.sin(phi)],
             [v[2] * v[0] * (1 - math.cos(phi)) - v[1] * math.sin(phi),
              v[2] * v[1] * (1 - math.cos(phi)) + v[0] * math.sin(phi),
              v[2] * v[2] * (1 - math.cos(phi)) + math.cos(phi)]]
        R = np.around(R, decimals = 6)
        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return v,phi,R
