#ifndef _KVSERVICE_HPP_
#define _KVSERVICE_HPP_

#include <iostream>
#include <sstream>
#include <string>
#include <vector> // Added for std::vector
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
#include "kvserverstub.hpp"

using namespace std;

class KVServiceServer: public Service {
    // system state
    string DBMFileName;
#ifdef __APPLE__
    DBM * dataFile = NULL;
#else
    GDBM_FILE dataFile;
#endif

    // rpc specific values
    static const uint32_t magic = 'E477';
    static const uint32_t version1x = 0x0100;

    // network management
    int sockfd;
    SDStub* sdstub;
    in_port_t PORT;
    string nickname;
    static const int MAXMSG = 1400;
    uint8_t udpMessage[MAXMSG];
    vector<KVServerStub> backupStubs; // For primary server to communicate with backups

    // New member variables for backup and primary configuration
    vector<string> backupIPs; // Names of backup servers
    vector<in_port_t> backupPorts; // Ports of backup servers
    string primaryName; // Name of the primary server (for backups)
   // bool isPrimary = false; // Flag to indicate if this server is the primary

    void callMethodVersion1(E477KV::kvRequest &receivedMsg, E477KV::kvResponse &replyMsg);
    bool kvPut(int key, const uint8_t * value, uint16_t vlen);
    kvGetResult kvGet(int key);
    bool registerService();
    
public:
    KVServiceServer(string name, weak_ptr<Node> p) : Service(name + ".KV_RPC", p), sdstub(new SDStub(name)) {}
    ~KVServiceServer() = default;

    void start();
    void stop();
    
    void setDBMFileName(string name) { DBMFileName = "data/" + name; }
    void setNickname(string newNickname) { nickname = newNickname; }
    void setPort(in_port_t newPORT) { PORT = newPORT; }

    // Methods to configure backups and the primary server
    bool addBackupName(string name);
    bool addBackupPort(in_port_t port);
    void setPrimaryServer(const string& name);
    void setIsPrimary(bool value) { isPrimary = value; } // Method to set the server as primary or backup
    bool isPrimary = false; // Flag to indicate if this server is the primary
};

#endif // _KVSERVICE_HPP_
