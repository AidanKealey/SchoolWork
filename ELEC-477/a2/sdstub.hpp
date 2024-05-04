#ifndef SDSTUB_HPP
#define SDSTUB_HPP

#include "network.hpp"
#include "DNS_custom.hpp"
#include "E477DirectoryService.pb.h"
#include "dumpHex.hpp"

class SDStub {
    // system management values
    string nickname;
    bool ready;
    string dsServerAddr = "10.0.0.1";
    string serverName = "Directory1";

    // rpc values
    static const uint32_t magic = 'E477';
    static const uint32_t version1x = 0x0200;
    atomic<uint32_t> serial;

    // network management
    int sockfd;
    in_port_t PORT = 1515;
    static const int MAXMSG = 1400;
    uint8_t udpMessage[MAXMSG];
    struct sockaddr_in serveraddr;

    bool init();

    public:
        SDStub(string nickname) : nickname(nickname), ready(false), serial(1) {}
        ~SDStub() {}

        bool registerNewServer(const string &nickname, struct directoryRecord aRecord);
        struct directoryRecord searchForService(string serviceName);
        bool deleteServer(string serviceName);
};

#endif