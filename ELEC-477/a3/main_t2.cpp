/*+
 *  File:   main.cpp
 *
 *  Purpose:
 *      This module is the start driver for several of the ELEC 477 assignments.
 *      It initializes the
-*/
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>
#include <string>


#include "network.hpp"
#include "kvserver.hpp"
#include "kvclient1.hpp"

#include "sdserver.hpp"
#include "sdstub.hpp"
#include "DNS_custom.hpp"

extern std::map<std::thread::id,shared_ptr<Node>> nodes;
extern std::map<std::thread::id,string> names;

int main(int argc, char * argv[]){
    // handle command line arguments...
    int res = network_init(argc, argv);
    std::stringstream ss;

    // start all of the servers first. This will let them get up
    // and running before the client attempts to communicste
    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: starting servers" << std::endl;

    std::cout << "Main: starting directory server" << std::endl;

    shared_ptr<SDServer> sdServer = make_shared<SDServer>("Directory1");

    sdServer->setAddress("10.0.0.1");
    sdServer->init();
    sdServer->startServices();

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::cout << "Main: starting other servers" << std::endl;
    // make shared broken?
    shared_ptr<KVServer> kvServer1 = make_shared<KVServer>("server1");
    shared_ptr<KVServer> kvServer2 = make_shared<KVServer>("server2");

    kvServer1->setAddress("10.0.0.12");
    kvServer1->setDBMFileName("server1");
    kvServer1->setNickname("snowRemoval");
    kvServer1->setPort(1515);
    kvServer1->init();
    kvServer1 -> startServices();

    kvServer2->setAddress("10.0.0.13");
    kvServer2->setDBMFileName("server2");
    kvServer2->setNickname("grassCutting");
    kvServer2->setPort(5151);
    kvServer2->init();
    kvServer2 -> startServices();

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: init client" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    shared_ptr<KVClient1> kvClient1 = make_shared<KVClient1>("kvclient1");
    shared_ptr<KVClient1> kvClient2 = make_shared<KVClient1>("kvclient2");
    kvClient1->setAddress("10.0.0.14");
    kvClient1->setServerName("snowRemoval");
    kvClient1 -> init();
    kvClient2->setAddress("10.0.0.15");
    kvClient2->setServerName("grassCutting");
    kvClient2 -> init();
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: starting client" << std::endl;
    vector<shared_ptr<thread>> clientThreads;
    {
        // need a scope for the lock guard. 
        // if this doesn't work put it in a function
        std::lock_guard<std::mutex> guard(nodes_mutex);

        shared_ptr<thread> t1 = make_shared<thread>([kvClient1](){
            kvClient1 -> execute();
        });

        clientThreads.push_back(t1);
        nodes.insert(make_pair(t1->get_id(), kvClient1));
        names.insert(make_pair(t1->get_id(),"kvclient1"));
        
        shared_ptr<thread> t2 = make_shared<thread>([kvClient2](){
            kvClient2 -> execute();
        });

        clientThreads.push_back(t2);
        nodes.insert(make_pair(t2->get_id(), kvClient2));
        names.insert(make_pair(t2->get_id(),"kvclient2"));

    }

    // wait for clients to finish
    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: waiting for clients to finish" << std::endl;
    vector<shared_ptr<thread>>::iterator thit;
    for (thit = clientThreads.begin(); thit != clientThreads.end(); thit++){
        shared_ptr<thread> tmp = *thit;
        tmp->join();
    }

    // when clients finish, shut down the servers
    // TODO - combine into node stop? that is node stop should
    // shut down all services and the client.
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: calling stop services on servers" << std::endl;
    kvServer1 -> stopServices();
    kvServer2 -> stopServices();

    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: waiting for threads to complete" << std::endl;
    // wait for all server threads
    kvServer1 -> waitForServices();
    kvServer2 -> waitForServices();

    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: stopping directory" << std::endl;
    sdServer -> stopServices();
    sdServer -> waitForServices();
    
    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}

