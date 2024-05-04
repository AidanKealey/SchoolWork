#ifndef __KV_SERVER_STUB_HPP_
#define __KV_SERVER_STUB_HPP_

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

class KVServerStub{
    string name;
    bool ready;
    static const uint32_t magic = 'E477';
    static const uint32_t version1x = 0x0100;
    atomic<uint32_t> serial;
    string backupIP;
    string IPaddress;
    in_port_t backupPort;
    SDStub* sdstub;



    // network management
    int sockfd;
    in_port_t serverPORT;
    string serverName;
    static const int MAXMSG = 1400;
    uint8_t udpMessage[MAXMSG];
    struct sockaddr_in servaddr;
    bool init();

public:
    ~KVServerStub() = default;
    void shutdown();
    bool kvPut(int32_t key, const uint8_t *value, uint16_t vlen);
    KVServerStub(string name, string backupIP, in_port_t backupPort);
    KVServerStub(const KVServerStub& kvserverstub);

    
};

#endif

