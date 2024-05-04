#ifndef __SERVICE_HPP__
#define __SERVICE_HPP__

#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <map>
#include <atomic>
#include <gdbm.h>

#include "dumpHex.hpp"
#include "network.hpp"
#include "rpc.pb.h"

using namespace std;

class RPCService : public Service {
    int sockfd;
    in_port_t PORT = 8080;
    static const int MAXMSG = 1400;
    uint8_t udpMessage[MAXMSG];
    uint8_t udpReply[MAXMSG];
    struct sockaddr_in cliaddr;

    private:
        string databaseName;
        GDBM_FILE database;
        //DBM* ndbmDatabase;

    public:
        RPCService();
        RPCService(string nodeName, weak_ptr<Node> p):Service(nodeName + ".a1_service",p) { }
        ~RPCService() {
            stop();
        }

        void start();
        void stop();

        void setDatabaseName(const string& name);

        bool initDatabase();
        void closeDatabase();

        void dispatch(const uint8_t* udpMessage, size_t messageLength, const sockaddr_in& clientAddress, socklen_t clientAddressLength);
        bool Put(int32_t key, const uint8_t * value, uint16_t value_len);
        RPC::kvGetResponse Get(int32_t key);
};

#endif
