#ifndef __KVSTUB_HPP_
#define __KVSTUB_HPP_

#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <atomic>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>

#ifdef __APPLE__
#define MSG_CONFIRM 0
#endif

#include "KVrpc.h"
#include "E477KV.pb.h"
#include "dumpHex.hpp"
#include "sdstub.hpp"
#include "DNS_custom.hpp"


using namespace std;

class KVServiceStub{
    // system managment values
    string name;
    bool ready;
    // string serverAddrString;
    // string serverName;

    // rpc specific values
    // version 1.0
    static const uint32_t magic = 'E477';
    static const uint32_t version1x = 0x0100;
    atomic<uint32_t> serial;

    // network management
    int sockfd;
    // static const in_port_t PORT = 8080;
    in_port_t serverPORT;
    string serverName;
    static const int MAXMSG = 1400;
    uint8_t udpMessage[MAXMSG];
    struct sockaddr_in servaddr;
    SDStub* sdstub;
    directoryRecord searchRecord;

    bool init();

public:
    KVServiceStub(string name ): name(name),ready(false), serial(1), sdstub(new SDStub(name)) {}
    ~KVServiceStub(){
        // shutdown();
        if (sdstub != nullptr) {
            delete sdstub;
            sdstub = nullptr;
        }
    }
    void shutdown();
    // void setServerAddress(string address){
    //     serverAddrString = address;
    // }
    void setServerName(string name){
        serverName = name;
    }
    bool kvPut(int32_t, const uint8_t* value, uint16_t vlen);
    kvGetResult kvGet(int32_t);
};

#endif

