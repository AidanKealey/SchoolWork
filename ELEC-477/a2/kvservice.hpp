

#ifndef _KVSERVICE_HPP_
#define _KVSERVICE_HPP_

#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>

#ifdef __APPLE__
#include <ndbm.h>
#else
#include <gdbm.h>
#endif


#include "KVrpc.h"
#include "E477KV.pb.h"
#include "dumpHex.hpp"
#include "network.hpp"
#include "sdstub.hpp"
#include "DNS_custom.hpp"

using namespace std;


class KVServiceServer: public Service{
    // system state
    string DBMFileName;
#ifdef __APPLE__
    DBM * dataFile = NULL;
#else
    GDBM_FILE dataFile;
#endif
    
    // rpc specific values
    // version 1.0
    static const uint32_t magic = 'E477';
    static const uint32_t version1x = 0x0100;

    // network management
    int sockfd;
    // in_port_t PORT = 8080;
    SDStub* sdstub;
    in_port_t PORT;
    string nickname;
    static const int MAXMSG = 1400;
    uint8_t udpMessage[MAXMSG];
    
    void callMethodVersion1(E477KV::kvRequest &receivedMsg, E477KV::kvResponse &replyMsg);
    bool kvPut(int key, const uint8_t * value, uint16_t vlen);
    kvGetResult kvGet(int key);
    bool registerService();
    bool deleteService();
    
public:
    KVServiceServer(string name, weak_ptr<Node> p) : Service(name + ".KV_RPC" ,p), sdstub(new SDStub(name)){}
    ~KVServiceServer(){
        //stop();

        if (sdstub != nullptr) {
            delete sdstub;
            sdstub = nullptr;
        }
        
#ifdef __APPLE__
        if (dataFile != nullptr) {
            dbm_close(dataFile);
            dataFile = nullptr;
        }
#else
        if (dataFile != nullptr) {
            gdbm_close(dataFile);
            dataFile = nullptr;
	}
#endif
        
    }
    void start();
    void stop();
    
    void setDBMFileName(string name) {DBMFileName = "data/" + name;}
    void setNickname(string newNickname) {nickname = newNickname;}
    void setPort(in_port_t newPORT) {PORT = newPORT;}
};

#endif
