#ifndef KVSERVER_HPP
#define KVSERVER_HPP

#include "network.hpp"
#include "kvservice.hpp"

class KVServer: public Node {
private:
    shared_ptr<KVServiceServer> kvService;

public:
    KVServer(string nodeName);
    ~KVServer(){}

    void setDBMFileName(string name);
    void setNickname(string nickname);
    void setPort(in_port_t PORT);
    void setIsPrimary(bool isPrimary); // New
    void addBackupName(string address); // New
    void addBackupPort(in_port_t port); // New
    void setPrimaryServer(string name); // New
};

#endif // KVSERVER_HPP
