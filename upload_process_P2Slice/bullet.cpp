#include<stdio.h>
#include<stdlib.h>
// #include "btBulletDynamicsCommon.h"

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#include <string>
#include <sstream>
#include <vector>
#include <iterator>

#include <dirent.h>

#include "BulletCollision/CollisionShapes/btTriangleMesh.h"

#include "btBulletDynamicsCommon.h"
#include "LinearMath/btVector3.h"
#include "LinearMath/btAlignedObjectArray.h"

#include "Bullet3Common/b3FileUtils.h"
#include "BulletCollision/CollisionShapes/btTriangleMesh.h"
// #include "BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h" # only for static object
#include "BulletCollision/Gimpact/btGImpactShape.h"
#include "BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h"

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
            *(result++) = item;
        }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

btTriangleMesh* trimeshFromObj(string fileName) {
    string line;
    ifstream myfile (fileName.c_str());
    if (myfile.is_open())
    {
        btTriangleMesh* trimesh = new btTriangleMesh();
        vector< vector<double> > vertices;

        while ( getline (myfile,line) )
        {
            if (line[0] == 'v')
            {
                vector<string> v = split(line, ' ');
                double x = atof(v[1].c_str());
                double y = atof(v[2].c_str());
                double z = atof(v[3].c_str());
                double v_array[3] = {x, y, z};
                vector<double> v_vector;
                v_vector.push_back(x);
                v_vector.push_back(y);
                v_vector.push_back(z);
                vertices.push_back(v_vector);
                // printf("v %f %f %f \n", x, y, z);
                // printf("v %f %f %f \n", v_array[0], v_array[1], v_array[2]);
            } else if (line[0] == 'f')
            {
                vector<string> f = split(line, ' ');
                vector<double> f0 = vertices[atoi(f[1].c_str()) - 1];
                vector<double> f1 = vertices[atoi(f[2].c_str()) - 1];
                vector<double> f2 = vertices[atoi(f[3].c_str()) - 1];

                btVector3 v0(f0[0], f0[1], f0[2]);
                btVector3 v1(f1[0], f1[1], f1[2]);
                btVector3 v2(f2[0], f2[1], f2[2]);

                trimesh->addTriangle(v0, v1, v2);
            }
        }

        printf("length of vertices %ld \n", vertices.size());
        myfile.close();
        return trimesh;
    }
    else cout << "Unable to open file";
}

bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
            return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
        } else {
                return false;
            }
}

int main (int argc, char* argv[]) {
    vector< string > fileNames;
    // char* directoryName;
    if(argc==1)
    {
        // printf("\nNo Extra Command Line Argument Passed Other Than Program Name\n");
        // printf("\nError\n");
        return 1;
    }
    else if(argc>=2)
    {
        printf("\nNumber Of Arguments Passed: %d",argc);
        printf("\n----Following Are The Command Line Arguments Passed----");

        for(int counter=1;counter<argc;counter++)
        {
            printf("\nargv[%d]: %s\n",counter,argv[counter]);
            fileNames.push_back(argv[counter]);
            // directoryName = argv[1]
        }
        // directoryName = argv[1];
    }

    // DIR *dir;
    // struct dirent *ent;

    // if ((dir = opendir("./helicopter")) != NULL) {
    // if ((dir = opendir(directoryName)) != NULL) {
        // [> print all the files and directories within directory <]
        // while ((ent = readdir (dir)) != NULL) {
            // string fileName;
            // fileName = directoryName + string(ent->d_name);
            // cout << fileName << "\n";
            // if (hasEnding(fileName, ".obj"))
            // {
                // fileNames.push_back(fileName);
            // }
        // }
        // closedir (dir);
    // }

    btBroadphaseInterface* broadphase = new btDbvtBroadphase();
    btDefaultCollisionConfiguration* collisionConfiguration = new btDefaultCollisionConfiguration();
    btCollisionDispatcher* dispatcher = new btCollisionDispatcher(collisionConfiguration);
    btGImpactCollisionAlgorithm::registerAlgorithm(dispatcher);
    btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver;
    btDiscreteDynamicsWorld* dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration);
    dynamicsWorld->setGravity(btVector3(0, 0, -10));

    vector<btRigidBody*> rigidBodies;
    std::cout << "number of fileNames" << fileNames.size() << std::endl;

    int counter = 0;
    for(std::vector<string>::iterator it = fileNames.begin(); it != fileNames.end(); ++it, ++counter) {

        std::cout << "in fileNames" <<std::endl;
        std::cout << *it <<std::endl;
        const string thisFileName = *it;
        btTriangleMesh* trimesh = trimeshFromObj(thisFileName);
        btGImpactMeshShape* shapeNew = new btGImpactMeshShape(trimesh);
        shapeNew->updateBound(); // important
        shapeNew->setMargin(0.001);
        btCollisionShape* fallShape = shapeNew;
        btDefaultMotionState* fallMotionState =
            new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0, 0)));

        btScalar mass = 1;
        if (counter == 0) {
            mass = 0;
        }

        btVector3 fallInertia(0, 0, 0);
        fallShape->calculateLocalInertia(mass, fallInertia);
        btRigidBody::btRigidBodyConstructionInfo fallRigidBodyCI(mass, fallMotionState, fallShape, fallInertia);
        btRigidBody* fallRigidBody = new btRigidBody(fallRigidBodyCI);
        dynamicsWorld->addRigidBody(fallRigidBody);

        rigidBodies.push_back(fallRigidBody);
    }

    std::cout << rigidBodies.size() << std::endl;
    if (rigidBodies.size() < 2) {
        return 1; // there is less than 1 rigitBodies
    }

    // btCollisionShape* groundShape = new btStaticPlaneShape(btVector3(0, 1, 0), 1);
    // btDefaultMotionState* groundMotionState = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, -1, 0)));
    // btRigidBody::btRigidBodyConstructionInfo
        // groundRigidBodyCI(0, groundMotionState, groundShape, btVector3(0, 0, 0));
    // btRigidBody* groundRigidBody = new btRigidBody(groundRigidBodyCI);
    // dynamicsWorld->addRigidBody(groundRigidBody);

    cout << "start\n";
    for (int i = 0; i < 300; i++) {
        dynamicsWorld->stepSimulation(1 / 60.f, 10);
        int p_counter = 0;
        for(std::vector<btRigidBody*>::iterator i = rigidBodies.begin(); i != rigidBodies.end(); ++i, ++p_counter) {
            // std::cout << *i <<std::endl;
            btTransform trans;
            (*i)->getMotionState()->getWorldTransform(trans);
            float x = trans.getOrigin().getX();
            float y = trans.getOrigin().getY();
            float z = trans.getOrigin().getZ();
            printf("%d x %f y %f z %f \n", p_counter, x, y, z);

            if (x != 0. || y != 0) {
                return 1;
            }
        }
    }
    cout << "end\n";
    return 0;
}
