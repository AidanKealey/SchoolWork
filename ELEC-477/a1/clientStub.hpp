#ifndef __CLIENT_STUB_HPP__
#define __CLIENT_STUB_HPP__

#include "rpc.pb.h" 

#include <string>
#include <memory>
#include <netinet/in.h>
#include <arpa/inet.h>

using namespace std;

class ClientStub {
    private:
        string serverAddress;
        int sockfd; 
        bool isInitialized = false; 
        static constexpr in_port_t PORT = 8080;
        static const int MAXMSG = 1400; 
        uint8_t udpMessage[MAXMSG];
        uint8_t udpReply[MAXMSG];

    public:
        ClientStub() = default;
        ~ClientStub();

        void setServerAddress(const string& addr);
        void initializeNetworkConnection();
        void shutdownNetworkConnection();

        bool kvPut(int32_t key, const string& value);
        std::pair<bool, std::string> kvGet(int32_t key);
};

#endif
