#ifndef SDSERVICE_HPP
#define SDSERVICE_HPP

#include "E477DirectoryService.pb.h"
#include "dumpHex.hpp"
#include "network.hpp" 
#include "DNS_custom.hpp"

class SDService : public Service {

    static const uint32_t magic = 'E477';
    static const uint32_t version1x = 0x0200;

    int sockfd;
    in_port_t PORT = 1515;
    static const int MAXMSG = 1400;
    uint8_t udpMessage[MAXMSG];

    unordered_map<string, struct directoryRecord> directory;

    void queryDirectory(E477DirectoryService::DSRequest &receivedMsg, E477DirectoryService::DSResponse &replyMsg);
    bool registerServer(const string &name, struct directoryRecord aRecord);
    struct directoryRecord searchForService(string nickname);
    bool deleteServer(string nickname);

    public:
        SDService(string name, weak_ptr<Node> p) : Service(name, p) {}
        ~SDService() {}

        void start();
        void stop();

};

#endif