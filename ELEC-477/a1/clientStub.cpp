#include "clientStub.hpp"
#include "rpc.pb.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <cerrno>
#include <cstring> 

using namespace std;

ClientStub::~ClientStub() {
    if (isInitialized) {
        shutdownNetworkConnection();
    }
}


void ClientStub::setServerAddress(const string& addr) {
    serverAddress = addr;
}


void ClientStub::initializeNetworkConnection() {
    if (!isInitialized) {
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0) {
            cerr << "Error opening socket" << endl;
            exit(EXIT_FAILURE);
        }

        struct timeval timeout;
        timeout.tv_sec = 5; 
        timeout.tv_usec = 0; 

        if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout)) < 0) {
            cerr << "Failed to set socket receive timeout" << endl;
            close(sockfd);
            exit(EXIT_FAILURE);
        }

        isInitialized = true;
    }
}


void ClientStub::shutdownNetworkConnection() {
    close(sockfd);
    isInitialized = false;
}


bool ClientStub::kvPut(int32_t key, const string& value) {
    initializeNetworkConnection();
    cout << "ClientStub::kvPut - Sending to " << serverAddress << ":" << ClientStub::PORT << endl;

    struct sockaddr_in servaddr;
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(ClientStub::PORT);
    inet_pton(AF_INET, serverAddress.c_str(), &servaddr.sin_addr);

    RPC::rpcHeader header;
    header.set_magic_number(1);
    header.set_version(2);
    auto* putRequest = header.mutable_put_request();
    putRequest->set_key(key);
    putRequest->set_value(value);

    size_t messageSize = header.ByteSizeLong();
    uint8_t sendmessageBuffer[messageSize]; 

    if (!header.SerializeToArray(sendmessageBuffer, messageSize)) {
        cerr << "Failed to serialize kvPut request." << endl;
        return false;
    }
    
    cerr << "ClientStub::kvPut - Serialized message bytes: ";
    for (size_t i = 0; i < messageSize; ++i) {
        std::cerr << std::hex << (0xFF & static_cast<int>(sendmessageBuffer[i])) << " ";
    }
    
    cerr << std::dec << std::endl;

    cout << "ClientStub::kvPut - Serialized message size: " << messageSize << " bytes" << endl;

    cerr << "I am the client. I am sending a message to the server at address " << serverAddress << " and port " << ClientStub::PORT << ".\n";

    // Send the request
    if (sendto(sockfd, sendmessageBuffer, messageSize, 0, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
        cerr << "sendto failed" << endl;
        return false;
    }

    uint8_t responseBuffer[1024]; 
    socklen_t len = sizeof(servaddr);
    int n = recvfrom(sockfd, responseBuffer, sizeof(responseBuffer), 0, (struct sockaddr*)&servaddr, &len);
    if (n < 0) {
        if (errno == EWOULDBLOCK) {
            cout << "client timed out" << endl;
        } else {
            cerr << "recvfrom failed: " << strerror(errno) << endl;
        }
        return false;
    }

    cout << "ClientStub::kvPut - Received " << n << " bytes in response." << endl;

    RPC::rpcHeader responseHeader;
    if (!responseHeader.ParseFromArray(responseBuffer, n)) {
        cerr << "Failed to parse kvPut response." << endl;
        return false;
    }

    if (responseHeader.has_put_response()) {
        return responseHeader.put_response().status();
    }

    return false;
}


pair<bool, string> ClientStub::kvGet(int32_t key) {
    cout << "Entering ClientStub::kvGet" << endl;
    initializeNetworkConnection();

    cout << "ClientStub::kvGet - Sending to " << serverAddress << ":" << PORT << endl;

    struct sockaddr_in servaddr;
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(PORT);
    inet_pton(AF_INET, serverAddress.c_str(), &servaddr.sin_addr);

    RPC::rpcHeader header;
    header.set_magic_number(1);
    header.set_version(2);
    auto* getRequest = header.mutable_get_request();
    getRequest->set_key(key);

    size_t messageSize = header.ByteSizeLong();
    uint8_t sendmessageBuffer[messageSize]; 

    if (!header.SerializeToArray(sendmessageBuffer, messageSize)) {
        cerr << "Failed to serialize kvGet request." << endl;
        return {false, ""};
    }

    cout << "ClientStub::kvGet - Serialized message bytes: ";
    for (size_t i = 0; i < messageSize; ++i) {
        cout << hex << (0xFF & static_cast<int>(sendmessageBuffer[i])) << " ";
    }
    cout << dec << endl;

    cout << "ClientStub::kvGet - Serialized message size: " << messageSize << " bytes" << endl;
    cout << "Sending kvGet request to server." << endl;

    cerr << "I am the client. I am sending a message to the server at address " << serverAddress << " and port " << ClientStub::PORT << ".\n";

    // Send the request
    if (sendto(sockfd, sendmessageBuffer, messageSize, 0, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
        cerr << "sendto failed for kvGet" << endl;
        return {false, ""};
    }

    uint8_t responseBuffer[1024];
    socklen_t len = sizeof(servaddr);
    int n = recvfrom(sockfd, responseBuffer, sizeof(responseBuffer), 0, (struct sockaddr*)&servaddr, &len);
    if (n < 0) {
        if (errno == EWOULDBLOCK) {
            cout << "client timed out" << endl;
        } else {
            cerr << "recvfrom failed for kvGet" << strerror(errno) << endl;
        }
        return {false, ""};
    }

    cerr << "I am the client. I received a response from the server at address " << serverAddress << " and port " << ClientStub::PORT << ".\n";
    cout << "Received response for kvGet, bytes: " << n << endl;

    RPC::rpcHeader responseHeader;
    if (!responseHeader.ParseFromArray(responseBuffer, n)) {
        cerr << "Failed to parse kvGet response." << endl;
        return {false, ""};
    }

    if (responseHeader.has_get_response()) {
        const auto& getResponse = responseHeader.get_response();
        if (getResponse.status()) {
            return {true, getResponse.value()};
        }
    }

    return {false, ""};
}
