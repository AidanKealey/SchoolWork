#include "service.hpp"
#include "rpc.pb.h"
#include <gdbm.h>
#include <netinet/in.h>  // For the AF_INET and sockaddr_in
#include <cstring>       // For memset
#include <iostream>      // For cerr
#include <sys/socket.h>  // For socket functions
#include <arpa/inet.h>   // For inet_ntop
#include <iomanip>       // For setw, setfill

#ifdef __APPLE__
#define MSG_CONFIRM 0
#endif

using namespace std;
using namespace string_literals;

#define close mclose
void mclose(int fd);

void RPCService::start() {
    cerr << "in RPCService::start" << endl;

    struct sockaddr_in servaddr, cliaddr;

    // Create a socket to receive messages
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Clear structure memory and set server address information
    memset(&servaddr, 0, sizeof(servaddr));
    memset(&cliaddr, 0, sizeof(cliaddr));

    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(PORT);

    // Bind the socket with the server address
    if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (!initDatabase()) {
        cerr << "Database initialization failed." << endl;
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    while (alive) {
        cerr << "waiting for call from client" << endl;

        socklen_t len = sizeof(cliaddr);
        int n = recvfrom(sockfd, udpMessage, MAXMSG, MSG_WAITALL, (struct sockaddr *)&cliaddr, &len);
        
        if (n >= 0) {
            cerr << "I am the server. I received a message from the client at address " << inet_ntoa(cliaddr.sin_addr) << " and port " << ntohs(cliaddr.sin_port) << ".\n";
        } else {
            perror("recvfrom error");
        }

        cerr << "server received " << n << " bytes." << endl;

        cerr << "Received bytes: ";
        for (int i = 0; i < min(n, 4); ++i) { 
            cerr << hex << (0xFF & static_cast<int>(udpMessage[i])) << " ";
        }
        cerr << dec << "\n";

        char clientStrBuffer[INET_ADDRSTRLEN];
        const char* clientstr = inet_ntop(AF_INET, &(cliaddr.sin_addr), clientStrBuffer, INET_ADDRSTRLEN);
        if (clientstr != nullptr) {
            cerr << "from address " << clientstr << endl;
        } else {
            perror("inet_ntop error");
        }

        dispatch(udpMessage, n, cliaddr, len);
    }

    closeDatabase();
    close(sockfd);
}



void RPCService::stop() {
    cerr << "in RPCService::stop" << endl;
    alive = false;
};


void RPCService::setDatabaseName(const string& name) {
    this->databaseName = name;
};


bool RPCService::initDatabase() {
    cerr << "in RPCService::initDatabase" << endl;
    database = gdbm_open(databaseName.c_str(), 0, GDBM_WRCREAT, 0666, nullptr);
    if (!database) {
        cerr << "Failed to open database: " << databaseName << endl;
        return false;
    }
    return true;
}


void RPCService::closeDatabase() {
    cerr << "in RPCService::closeDatabase" << endl;
    if (database) {
        gdbm_close(database);
        database = nullptr;
    }
}


void RPCService::dispatch(const uint8_t* udpMessage, size_t messageLength, const sockaddr_in& clientAddress, socklen_t clientAddressLength) {
    cerr << "in RPCService::dispatch" << endl;
    
    RPC::rpcHeader message;
    if (!message.ParseFromArray(udpMessage, messageLength)) {
        cerr << "Failed to parse incoming message." << endl;
        return;
    }

    cerr << "Extracted magic number: " << message.magic_number() << ", Expected: 1" << endl;

    if(message.magic_number() != 1) {
        cerr << "magic number incorrect" << endl;
        return;
    }

    cerr << "Extracted version: " << message.version() << ", Expected: 2" << endl;

    if(message.version() != 2) {
        cerr << "wrong version" << endl;
        return;
    }

    switch (message.message_case()) {
            // cerr << "i am here in the switch" << endl;
        case RPC::rpcHeader::kPutRequest: {
                cerr << "i am here in kPutRequest" << endl;
            const RPC::kvPutRequest& putRequest = message.put_request();
            const string& value = putRequest.value();
            bool status = Put(putRequest.key(), reinterpret_cast<const uint8_t*>(value.data()), value.length());

            RPC::kvPutResponse putResponse;
            putResponse.set_status(status);
            size_t respSize = putResponse.ByteSizeLong();
            if (!putResponse.SerializeToArray(udpReply, MAXMSG)) {
                cerr << "Serialization failed for putResponse" << endl;
                return;
            }
            
            cerr << "I am the server. I am sending a response to the client at address " << inet_ntoa(clientAddress.sin_addr) << " and port " << ntohs(clientAddress.sin_port) << ".\n";

            // Send response
            int servern = sendto(sockfd, udpReply, respSize, MSG_CONFIRM, reinterpret_cast<const struct sockaddr*>(&clientAddress), clientAddressLength);
            if (servern < 0) {
                perror("sendto failed");
            } else {
                cerr << "Server sent " << servern << " bytes in response to kvPutRequest" << endl;
                    cerr << "Server sent " << servern << " bytes in response to kvPutRequest. Data: ";
                for (int i = 0; i < servern; ++i) {
                    cerr << hex << setw(2) << setfill('0') << (0xff & static_cast<int>(udpReply[i])) << " ";
                }
                cerr << dec << endl;
            }
            
            break;
        }

        case RPC::rpcHeader::kGetRequest: {
            const RPC::kvGetRequest& getRequest = message.get_request();
            RPC::kvGetResponse getResponse = Get(getRequest.key());

            RPC::rpcHeader responseHeader;
            *responseHeader.mutable_get_response() = getResponse;

            size_t respSize = responseHeader.ByteSizeLong();
            if (respSize > MAXMSG) {
                cerr << "Serialized message size exceeds MAXMSG limit." << endl;
                return; 
            }
            uint8_t serializedResponse[respSize];

            if (!responseHeader.SerializeToArray(serializedResponse, respSize)) {
                cerr << "Serialization failed for getResponse" << endl;
                return;
            }

            int servern = sendto(sockfd, serializedResponse, respSize, 0,
                                reinterpret_cast<const struct sockaddr*>(&clientAddress), clientAddressLength);
            if (servern < 0) {
                perror("sendto failed");
            } else {
                cerr << "Server sent " << servern << " bytes in response to kvGetRequest" << endl;
            }
            break;
        }


        default: {
            cerr << "Unknown message type" << endl;
            break;
        }
    }
}


bool RPCService::Put(int32_t key, const uint8_t *value, uint16_t value_len) {
    std::cerr << "in RPCService::Put" << endl;
    if (!database) {
        cerr << "Database not open, cannot run put. Returning false." << endl;
        return false;
    }

    std::cerr << "i am here" << endl;

    datum databaseKey, databaseValue;
    databaseKey.dptr = reinterpret_cast<char*>(&key);
    databaseKey.dsize = sizeof(int32_t);
    databaseValue.dptr = reinterpret_cast<char*>(const_cast<uint8_t*>(value));
    databaseValue.dsize = value_len;

    std::cerr << "i am here 2" << endl;

    int storeResult = gdbm_store(database, databaseKey, databaseValue, GDBM_REPLACE);
    std::cerr << "i am here 3" << endl;
    if (storeResult != 0) {
        std::cerr << "Failed to store value in the database. GDBM store result: " << storeResult << ". Returning false." << endl;
        return false;
    }
    std::cerr << "Value successfully stored in the database." << endl;
    return true;
}


RPC::kvGetResponse RPCService::Get(int32_t key) {
    cerr << "in RPCService::Get" << endl;
    RPC::kvGetResponse response;
    response.set_status(false);

    if (!database) {
        cerr << "Database not open, cannot run get. Returning response with status false." << endl;
        return response;
    }

    datum databaseKey;
    databaseKey.dptr = reinterpret_cast<char*>(&key);
    databaseKey.dsize = sizeof(key); 

    datum result = gdbm_fetch(database, databaseKey);
    if (result.dptr != nullptr) {
        
        response.set_status(true);
        response.set_value(std::string(result.dptr, result.dsize));
        free(result.dptr);
    }

    return response;
}



