
#ifndef __KVCLIENT_HPP_
#define __KVCLIENT_HPP_
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <atomic>
#include <sys/socket.h>
#include <arpa/inet.h>

#ifdef __APPLE__
#define MSG_CONFIRM 0
#endif

#include "KVrpc.h"
#include "E477KV.pb.h"
#include "dumpHex.hpp"

#include "kvclientstub.hpp"
#include "network.hpp"

using namespace std;

class KVClient1: public Node {
    // for now one service stub
    KVServiceStub kvService;
    // this alive refers to client activities
    //std::atomic<bool> alive;

public:
    KVClient1(string name): Node(name), kvService(name){
    }
    ~KVClient1(){
        // kvService.shutdown();
        // stop();
    }
    void start();
    // void setServerAddress(string addr);
    void setServerName(string name);
};


#endif

