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

    kvServer1->setAddress("10.0.0.12");
    kvServer1->setDBMFileName("server1");
    kvServer1->setNickname("snow");
    kvServer1->setPort(1515);
    kvServer1->init();
    kvServer1 -> startServices();

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: init client" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    shared_ptr<KVClient1> kvClient = make_shared<KVClient1>("kvclient");
    kvClient->setAddress("10.0.0.14");
    kvClient->setServerName("snow");
    kvClient -> init();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: starting client" << std::endl;
    vector<shared_ptr<thread>> clientThreads;
    {
        // need a scope for the lock guard. 
        // if this doesn't work put it in a function
	std::lock_guard<std::mutex> guard(nodes_mutex);

	shared_ptr<thread> t = make_shared<thread>([kvClient](){
	    kvClient -> execute();
	});

	clientThreads.push_back(t);
	nodes.insert(make_pair(t->get_id(), kvClient));
	names.insert(make_pair(t->get_id(),"kvclient"));

    }

    // wait for clients to finish
    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: waiting for clients to finish" << std::endl;
    vector<shared_ptr<thread>>::iterator thit;
    for (thit = clientThreads.begin(); thit != clientThreads.end(); thit++){
        shared_ptr<thread> tmp = *thit;
        tmp->join();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: calling stop services on servers" << std::endl;
    kvServer1 -> stopServices();

    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: waiting for threads to complete" << std::endl;
    // wait for all server threads
    kvServer1 -> waitForServices();

    std::cout << "Main: ************************************" << std::endl;
    std::cout << "Main: stopping directory" << std::endl;
    sdServer -> stopServices();
    sdServer -> waitForServices();

    //std::this_thread::sleep_for(std::chrono::milliseconds(10000));

    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}

