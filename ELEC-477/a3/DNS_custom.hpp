#ifndef _DNS_CUSTOM_HPP_
#define _DNS_CUSTOM_HPP_

#include <string>
#include "network.hpp"

struct directoryRecord {
    string server;
    in_port_t port;
};

#endif