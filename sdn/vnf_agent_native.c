/*
 * Native persistent UDP VNF agent.
 *
 * It intentionally keeps the JSON FIFO protocol used by vnf_agent.py so the
 * controller and Mininet runtime do not need a second control path.  The data
 * path is a single resident poll loop with non-blocking UDP sockets and no
 * per-packet Python object allocation or hashing.
 */

#define _GNU_SOURCE

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define MAX_BINDINGS 1024
#define MAX_JSON_LINE 65536
#define MAX_PAYLOAD 65535
#define DEFAULT_PACKET_BURST 64
#define MAX_IO_BATCH 64
#define DEFAULT_DRAIN_TIMEOUT_MS 100.0
#define DEFAULT_DRAIN_IDLE_MS 5.0

typedef struct {
    int used;
    int receiver_fd;
    int sender_fd;
    int request_id;
    int stage;
    int vnf_type;
    int dscp;
    int draining;
    int drop_every;
    int processing_delay_us;
    uint64_t migration_epoch;
    uint64_t received;
    uint64_t forwarded;
    uint64_t dropped;
    double started_ms;
    double drain_requested_ms;
    double drain_deadline_ms;
    double drain_last_packet_ms;
    double drain_idle_ms;
    char next_host[256];
    int next_port;
    struct sockaddr_in next_addr;
    char ready_file[PATH_MAX];
    char stats_output[PATH_MAX];
    char drain_ack[PATH_MAX];
    char drain_ack_socket[108];
    char drain_ack_token[128];
} binding_t;

static volatile sig_atomic_t g_stopped = 0;

static void stop_signal(int signo) {
    (void)signo;
    g_stopped = 1;
}

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void skip_ws(const char **cursor) {
    while (**cursor && isspace((unsigned char)(*cursor)[0])) {
        (*cursor)++;
    }
}

static const char *json_value(const char *json, const char *key) {
    char needle[128];
    int n = snprintf(needle, sizeof(needle), "\"%s\"", key);
    if (n <= 0 || (size_t)n >= sizeof(needle)) {
        return NULL;
    }
    const char *p = strstr(json, needle);
    if (!p) {
        return NULL;
    }
    p += n;
    p = strchr(p, ':');
    if (!p) {
        return NULL;
    }
    p++;
    skip_ws(&p);
    return p;
}

static int json_string(const char *json, const char *key, char *out, size_t cap) {
    const char *p = json_value(json, key);
    if (!p || *p != '"' || cap == 0) {
        return -1;
    }
    p++;
    size_t used = 0;
    while (*p && *p != '"') {
        unsigned char ch = (unsigned char)*p++;
        if (ch == '\\' && *p) {
            ch = (unsigned char)*p++;
        }
        if (used + 1 < cap) {
            out[used++] = (char)ch;
        }
    }
    if (*p != '"') {
        return -1;
    }
    out[used] = '\0';
    return 0;
}

static int json_int(const char *json, const char *key, int *out) {
    const char *p = json_value(json, key);
    if (!p || !out) {
        return -1;
    }
    char *end = NULL;
    long value = strtol(p, &end, 10);
    if (end == p) {
        return -1;
    }
    *out = (int)value;
    return 0;
}

static int json_uint64(const char *json, const char *key, uint64_t *out) {
    const char *p = json_value(json, key);
    if (!p || !out) {
        return -1;
    }
    char *end = NULL;
    unsigned long long value = strtoull(p, &end, 10);
    if (end == p) {
        return -1;
    }
    *out = (uint64_t)value;
    return 0;
}

static int json_double(const char *json, const char *key, double *out) {
    const char *p = json_value(json, key);
    if (!p || !out) {
        return -1;
    }
    char *end = NULL;
    double value = strtod(p, &end);
    if (end == p) {
        return -1;
    }
    *out = value;
    return 0;
}

static int set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        return -1;
    }
    return 0;
}

static void write_text_atomic(const char *path, const char *value) {
    if (!path || !path[0]) {
        return;
    }
    char tmp[PATH_MAX];
    int n = snprintf(tmp, sizeof(tmp), "%s.%ld.tmp", path, (long)getpid());
    if (n <= 0 || (size_t)n >= sizeof(tmp)) {
        return;
    }
    FILE *file = fopen(tmp, "w");
    if (!file) {
        return;
    }
    fputs(value ? value : "", file);
    fclose(file);
    rename(tmp, path);
}

static void write_ready(const binding_t *binding) {
    write_text_atomic(binding->ready_file, "ready\n");
}

static void send_control_ack(
    const char *ack_socket,
    const char *ack_token,
    const char *operation,
    int request_id,
    int stage,
    int accepted,
    const char *error) {
    if (!ack_socket || !ack_socket[0] || !ack_token || !ack_token[0]) {
        return;
    }
    int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd < 0) {
        return;
    }
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    if (strlen(ack_socket) >= sizeof(address.sun_path)) {
        close(fd);
        return;
    }
    snprintf(address.sun_path, sizeof(address.sun_path), "%s", ack_socket);
    char payload[1024];
    int length;
    if (error && error[0]) {
        length = snprintf(
            payload,
            sizeof(payload),
            "{\"accepted\":false,\"ack_token\":\"%s\",\"error\":\"%s\"," 
            "\"operation\":\"%s\",\"ready_monotonic_ns\":%llu,"
            "\"request_id\":%d,\"stage\":%d,\"worker_id\":0}",
            ack_token,
            error,
            operation,
            (unsigned long long)(monotonic_ms() * 1000000.0),
            request_id,
            stage);
    } else {
        length = snprintf(
            payload,
            sizeof(payload),
            "{\"accepted\":%s,\"ack_token\":\"%s\",\"operation\":\"%s\"," 
            "\"ready_monotonic_ns\":%llu,\"request_id\":%d,\"stage\":%d,"
            "\"worker_id\":0}",
            accepted ? "true" : "false",
            ack_token,
            operation,
            (unsigned long long)(monotonic_ms() * 1000000.0),
            request_id,
            stage);
    }
    if (length > 0 && (size_t)length < sizeof(payload)) {
        sendto(
            fd,
            payload,
            (size_t)length,
            0,
            (struct sockaddr *)&address,
            sizeof(address));
    }
    close(fd);
}

static void send_unregister_ack(
    const char *ack_socket,
    const char *ack_token,
    int request_id,
    int stage,
    uint64_t received,
    uint64_t forwarded) {
    if (!ack_socket || !ack_socket[0] || !ack_token || !ack_token[0]) {
        return;
    }
    int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd < 0) {
        return;
    }
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    if (strlen(ack_socket) >= sizeof(address.sun_path)) {
        close(fd);
        return;
    }
    snprintf(address.sun_path, sizeof(address.sun_path), "%s", ack_socket);
    char payload[1024];
    int length = snprintf(
        payload,
        sizeof(payload),
        "{\"accepted\":true,\"ack_token\":\"%s\",\"forwarded\":%llu,"
        "\"operation\":\"unregister\",\"ready_monotonic_ns\":%llu,"
        "\"received\":%llu,\"request_id\":%d,\"stage\":%d,\"worker_id\":0}",
        ack_token,
        (unsigned long long)forwarded,
        (unsigned long long)(monotonic_ms() * 1000000.0),
        (unsigned long long)received,
        request_id,
        stage);
    if (length > 0 && (size_t)length < sizeof(payload)) {
        sendto(
            fd,
            payload,
            (size_t)length,
            0,
            (struct sockaddr *)&address,
            sizeof(address));
    }
    close(fd);
}

static void send_state_ack(
    const char *ack_socket,
    const char *ack_token,
    const char *operation,
    const binding_t *binding) {
    if (!ack_socket || !ack_socket[0] || !ack_token || !ack_token[0]) {
        return;
    }
    int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd < 0) {
        return;
    }
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    if (strlen(ack_socket) >= sizeof(address.sun_path)) {
        close(fd);
        return;
    }
    snprintf(address.sun_path, sizeof(address.sun_path), "%s", ack_socket);
    char payload[2048];
    int length = snprintf(
        payload,
        sizeof(payload),
        "{\"accepted\":true,\"ack_token\":\"%s\",\"dropped\":%llu,"
        "\"forwarded\":%llu,\"migration_epoch\":%llu,\"operation\":\"%s\","
        "\"ready_monotonic_ns\":%llu,\"received\":%llu,\"request_id\":%d,"
        "\"stage\":%d,\"state\":{\"dropped\":%llu,\"forwarded\":%llu,"
        "\"migration_epoch\":%llu,\"received\":%llu,"
        "\"schema\":\"udp_forwarder.v1\"},\"worker_id\":0}",
        ack_token,
        (unsigned long long)binding->dropped,
        (unsigned long long)binding->forwarded,
        (unsigned long long)binding->migration_epoch,
        operation,
        (unsigned long long)(monotonic_ms() * 1000000.0),
        (unsigned long long)binding->received,
        binding->request_id,
        binding->stage,
        (unsigned long long)binding->dropped,
        (unsigned long long)binding->forwarded,
        (unsigned long long)binding->migration_epoch,
        (unsigned long long)binding->received);
    if (length > 0 && (size_t)length < sizeof(payload)) {
        sendto(fd, payload, (size_t)length, 0,
               (struct sockaddr *)&address, sizeof(address));
    }
    close(fd);
}

static void send_update_ack(
    const char *ack_socket,
    const char *ack_token,
    const binding_t *binding) {
    if (!ack_socket || !ack_socket[0] || !ack_token || !ack_token[0]) {
        return;
    }
    int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd < 0) {
        return;
    }
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    if (strlen(ack_socket) >= sizeof(address.sun_path)) {
        close(fd);
        return;
    }
    snprintf(address.sun_path, sizeof(address.sun_path), "%s", ack_socket);
    char payload[1024];
    int length = snprintf(
        payload,
        sizeof(payload),
        "{\"accepted\":true,\"ack_token\":\"%s\",\"migration_epoch\":%llu,"
        "\"next_host\":\"%s\",\"next_port\":%d,\"operation\":\"update_next\","
        "\"ready_monotonic_ns\":%llu,\"request_id\":%d,\"stage\":%d,"
        "\"worker_id\":0}",
        ack_token,
        (unsigned long long)binding->migration_epoch,
        binding->next_host,
        binding->next_port,
        (unsigned long long)(monotonic_ms() * 1000000.0),
        binding->request_id,
        binding->stage);
    if (length > 0 && (size_t)length < sizeof(payload)) {
        sendto(fd, payload, (size_t)length, 0,
               (struct sockaddr *)&address, sizeof(address));
    }
    close(fd);
}

static void write_stats(const binding_t *binding) {
    char json[2048];
    double elapsed = (monotonic_ms() - binding->started_ms) / 1000.0;
    snprintf(
        json,
        sizeof(json),
        "{\"dropped\":%llu,\"elapsed_seconds\":%.6f,\"forwarded\":%llu,"
        "\"migration_epoch\":%llu,\"received\":%llu,\"request_id\":%d,\"runtime\":\"native_vnf_agent\","
        "\"stage\":%d,\"vnf_type\":%d}\n",
        (unsigned long long)binding->dropped,
        elapsed,
        (unsigned long long)binding->forwarded,
        (unsigned long long)binding->migration_epoch,
        (unsigned long long)binding->received,
        binding->request_id,
        binding->stage,
        binding->vnf_type);
    write_text_atomic(binding->stats_output, json);
}

static void write_drain_ack(const binding_t *binding, int timed_out) {
    char json[2048];
    double wait_ms = monotonic_ms() - binding->drain_requested_ms;
    snprintf(
        json,
        sizeof(json),
        "{\"drain_wait_ms\":%.6f,\"forwarded\":%llu,\"received\":%llu,"
        "\"request_id\":%d,\"stage\":%d,\"timed_out\":%s}\n",
        wait_ms,
        (unsigned long long)binding->forwarded,
        (unsigned long long)binding->received,
        binding->request_id,
        binding->stage,
        timed_out ? "true" : "false");
    write_text_atomic(binding->drain_ack, json);
}

static void send_drain_control_ack(const binding_t *binding, int timed_out) {
    if (!binding->drain_ack_socket[0] || !binding->drain_ack_token[0]) {
        return;
    }
    int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd < 0) {
        return;
    }
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    if (strlen(binding->drain_ack_socket) >= sizeof(address.sun_path)) {
        close(fd);
        return;
    }
    snprintf(address.sun_path, sizeof(address.sun_path), "%s", binding->drain_ack_socket);
    char payload[2048];
    double wait_ms = monotonic_ms() - binding->drain_requested_ms;
    int length = snprintf(
        payload,
        sizeof(payload),
        "{\"accepted\":true,\"ack_token\":\"%s\",\"drain_wait_ms\":%.6f,"
        "\"forwarded\":%llu,\"operation\":\"drain\",\"ready_monotonic_ns\":%llu,"
        "\"received\":%llu,\"request_id\":%d,\"stage\":%d,"
        "\"timed_out\":%s,\"worker_id\":0}",
        binding->drain_ack_token,
        wait_ms,
        (unsigned long long)binding->forwarded,
        (unsigned long long)(monotonic_ms() * 1000000.0),
        (unsigned long long)binding->received,
        binding->request_id,
        binding->stage,
        timed_out ? "true" : "false");
    if (length > 0 && (size_t)length < sizeof(payload)) {
        sendto(
            fd,
            payload,
            (size_t)length,
            0,
            (struct sockaddr *)&address,
            sizeof(address));
    }
    close(fd);
}

static void close_binding(binding_t *binding) {
    if (!binding->used) {
        return;
    }
    if (binding->receiver_fd >= 0) {
        close(binding->receiver_fd);
    }
    if (binding->sender_fd >= 0) {
        close(binding->sender_fd);
    }
    write_stats(binding);
    memset(binding, 0, sizeof(*binding));
    binding->receiver_fd = -1;
    binding->sender_fd = -1;
}

static int resolve_ipv4(const char *host, int port, struct sockaddr_in *out) {
    memset(out, 0, sizeof(*out));
    out->sin_family = AF_INET;
    out->sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &out->sin_addr) == 1) {
        return 0;
    }
    struct addrinfo hints;
    struct addrinfo *result = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    int rc = getaddrinfo(host, NULL, &hints, &result);
    if (rc != 0 || !result) {
        return -1;
    }
    *out = *(struct sockaddr_in *)result->ai_addr;
    out->sin_port = htons((uint16_t)port);
    freeaddrinfo(result);
    return 0;
}

static int register_binding(
    binding_t *binding,
    const char *line,
    int default_burst,
    int default_q0_burst) {
    int request_id = 0;
    int stage = 0;
    int listen_port = 0;
    int next_port = 0;
    int vnf_type = 0;
    int dscp = 0;
    int drop_every = 0;
    int processing_delay_us = 0;
    uint64_t restore_received = 0;
    uint64_t restore_forwarded = 0;
    uint64_t restore_dropped = 0;
    uint64_t migration_epoch = 0;
    char next_host[sizeof(binding->next_host)] = {0};
    char ready_file[sizeof(binding->ready_file)] = {0};
    char stats_output[sizeof(binding->stats_output)] = {0};
    (void)default_burst;
    (void)default_q0_burst;
    if (json_int(line, "request_id", &request_id) < 0 ||
        json_int(line, "stage", &stage) < 0 ||
        json_int(line, "listen_port", &listen_port) < 0 ||
        json_int(line, "next_port", &next_port) < 0 ||
        json_string(line, "next_host", next_host, sizeof(next_host)) < 0) {
        fprintf(stderr, "native VNF agent: malformed register command\n");
        return -1;
    }
    json_string(line, "ready_file", ready_file, sizeof(ready_file));
    json_string(line, "stats_output", stats_output, sizeof(stats_output));
    json_int(line, "dscp", &dscp);
    json_int(line, "vnf_type", &vnf_type);
    json_int(line, "drop_every", &drop_every);
    json_int(line, "processing_delay_us", &processing_delay_us);
    json_uint64(line, "restore_received", &restore_received);
    json_uint64(line, "restore_forwarded", &restore_forwarded);
    json_uint64(line, "restore_dropped", &restore_dropped);
    json_uint64(line, "migration_epoch", &migration_epoch);
    if (listen_port <= 0 || listen_port > 65535 || next_port <= 0 || next_port > 65535) {
        return -1;
    }
    binding->receiver_fd = socket(AF_INET, SOCK_DGRAM, 0);
    binding->sender_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (binding->receiver_fd < 0 || binding->sender_fd < 0) {
        close_binding(binding);
        return -1;
    }
    int yes = 1;
    setsockopt(binding->receiver_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    int receive_buffer = 4 * 1024 * 1024;
    setsockopt(binding->receiver_fd, SOL_SOCKET, SO_RCVBUF, &receive_buffer, sizeof(receive_buffer));
    dscp = dscp < 0 ? 0 : (dscp > 63 ? 63 : dscp);
    int tos = dscp << 2;
    setsockopt(binding->sender_fd, IPPROTO_IP, IP_TOS, &tos, sizeof(tos));
    if (set_nonblocking(binding->receiver_fd) < 0 ||
        set_nonblocking(binding->sender_fd) < 0) {
        close_binding(binding);
        return -1;
    }
    struct sockaddr_in listen_addr;
    memset(&listen_addr, 0, sizeof(listen_addr));
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    listen_addr.sin_port = htons((uint16_t)listen_port);
    if (bind(binding->receiver_fd, (struct sockaddr *)&listen_addr, sizeof(listen_addr)) < 0 ||
        resolve_ipv4(next_host, next_port, &binding->next_addr) < 0) {
        close_binding(binding);
        return -1;
    }
    binding->used = 1;
    binding->request_id = request_id;
    binding->stage = stage;
    binding->vnf_type = vnf_type;
    binding->dscp = dscp;
    binding->drop_every = drop_every > 0 ? drop_every : 0;
    binding->processing_delay_us = processing_delay_us > 0 ? processing_delay_us : 0;
    binding->received = restore_received;
    binding->forwarded = restore_forwarded;
    binding->dropped = restore_dropped;
    binding->migration_epoch = migration_epoch;
    binding->started_ms = monotonic_ms();
    binding->next_port = next_port;
    snprintf(binding->next_host, sizeof(binding->next_host), "%s", next_host);
    snprintf(binding->ready_file, sizeof(binding->ready_file), "%s", ready_file);
    snprintf(binding->stats_output, sizeof(binding->stats_output), "%s", stats_output);
    write_ready(binding);
    return 0;
}

static binding_t *find_binding(binding_t *bindings, int request_id, int stage) {
    for (int i = 0; i < MAX_BINDINGS; i++) {
        if (bindings[i].used && bindings[i].request_id == request_id && bindings[i].stage == stage) {
            return &bindings[i];
        }
    }
    return NULL;
}

static binding_t *free_binding(binding_t *bindings) {
    for (int i = 0; i < MAX_BINDINGS; i++) {
        if (!bindings[i].used) {
            return &bindings[i];
        }
    }
    return NULL;
}

static void handle_command(
    binding_t *bindings,
    const char *line,
    int packet_burst,
    int q0_packet_burst,
    double drain_timeout_ms,
    double drain_idle_ms) {
    char operation[32] = {0};
    if (json_string(line, "operation", operation, sizeof(operation)) < 0) {
        fprintf(stderr, "native VNF agent: command without operation\n");
        return;
    }
    if (strcmp(operation, "shutdown") == 0) {
        g_stopped = 1;
        return;
    }
    int request_id = 0;
    int stage = 0;
    if (json_int(line, "request_id", &request_id) < 0 || json_int(line, "stage", &stage) < 0) {
        return;
    }
    char ack_socket[108] = {0};
    char ack_token[128] = {0};
    json_string(line, "ack_socket", ack_socket, sizeof(ack_socket));
    json_string(line, "ack_token", ack_token, sizeof(ack_token));
    if (strcmp(operation, "register") == 0) {
        if (find_binding(bindings, request_id, stage)) {
            fprintf(stderr, "native VNF agent: duplicate binding %d/%d\n", request_id, stage);
            send_control_ack(
                ack_socket, ack_token, operation, request_id, stage, 0, "duplicate_binding");
            return;
        }
        binding_t *slot = free_binding(bindings);
        if (!slot) {
            fprintf(stderr, "native VNF agent: no free binding for %d/%d\n", request_id, stage);
            send_control_ack(
                ack_socket, ack_token, operation, request_id, stage, 0, "no_free_binding");
            return;
        }
        if (register_binding(slot, line, packet_burst, q0_packet_burst) < 0) {
            fprintf(stderr, "native VNF agent: register failed for %d/%d: %s\n", request_id, stage, strerror(errno));
            send_control_ack(
                ack_socket, ack_token, operation, request_id, stage, 0, "registration_failed");
            return;
        }
        send_control_ack(
            ack_socket, ack_token, operation, request_id, stage, 1, NULL);
        return;
    }
    binding_t *binding = find_binding(bindings, request_id, stage);
    if (!binding) {
        send_control_ack(
            ack_socket, ack_token, operation, request_id, stage, 0, "missing_binding");
        return;
    }
    if (strcmp(operation, "snapshot") == 0) {
        send_state_ack(ack_socket, ack_token, operation, binding);
        return;
    }
    if (strcmp(operation, "restore") == 0) {
        uint64_t received = 0;
        uint64_t forwarded = 0;
        uint64_t dropped = 0;
        uint64_t migration_epoch = 0;
        json_uint64(line, "restore_received", &received);
        json_uint64(line, "restore_forwarded", &forwarded);
        json_uint64(line, "restore_dropped", &dropped);
        json_uint64(line, "migration_epoch", &migration_epoch);
        if (migration_epoch < binding->migration_epoch) {
            send_control_ack(
                ack_socket, ack_token, operation, request_id, stage, 0,
                "stale_migration_epoch");
            return;
        }
        if (received > binding->received) binding->received = received;
        if (forwarded > binding->forwarded) binding->forwarded = forwarded;
        if (dropped > binding->dropped) binding->dropped = dropped;
        binding->migration_epoch = migration_epoch;
        send_state_ack(ack_socket, ack_token, operation, binding);
        return;
    }
    if (strcmp(operation, "restore_delta") == 0) {
        uint64_t received = 0;
        uint64_t forwarded = 0;
        uint64_t dropped = 0;
        uint64_t migration_epoch = 0;
        json_uint64(line, "delta_received", &received);
        json_uint64(line, "delta_forwarded", &forwarded);
        json_uint64(line, "delta_dropped", &dropped);
        json_uint64(line, "migration_epoch", &migration_epoch);
        if (migration_epoch < binding->migration_epoch) {
            send_control_ack(
                ack_socket, ack_token, operation, request_id, stage, 0,
                "stale_migration_epoch");
            return;
        }
        binding->received += received;
        binding->forwarded += forwarded;
        binding->dropped += dropped;
        binding->migration_epoch = migration_epoch;
        send_state_ack(ack_socket, ack_token, operation, binding);
        return;
    }
    if (strcmp(operation, "update_next") == 0) {
        char next_host[sizeof(binding->next_host)] = {0};
        int next_port = 0;
        uint64_t migration_epoch = binding->migration_epoch;
        json_uint64(line, "migration_epoch", &migration_epoch);
        if (migration_epoch < binding->migration_epoch) {
            send_control_ack(
                ack_socket, ack_token, operation, request_id, stage, 0,
                "stale_migration_epoch");
            return;
        }
        if (json_string(line, "next_host", next_host, sizeof(next_host)) < 0 ||
            json_int(line, "next_port", &next_port) < 0 ||
            next_port <= 0 || next_port > 65535 ||
            resolve_ipv4(next_host, next_port, &binding->next_addr) < 0) {
            send_control_ack(
                ack_socket, ack_token, operation, request_id, stage, 0,
                "invalid_next_endpoint");
            return;
        }
        snprintf(binding->next_host, sizeof(binding->next_host), "%s", next_host);
        binding->next_port = next_port;
        binding->migration_epoch = migration_epoch;
        send_update_ack(ack_socket, ack_token, binding);
        return;
    }
    if (strcmp(operation, "update_impairment") == 0) {
        int drop_every = binding->drop_every;
        int processing_delay_us = binding->processing_delay_us;
        json_int(line, "drop_every", &drop_every);
        json_int(line, "processing_delay_us", &processing_delay_us);
        binding->drop_every = drop_every > 0 ? drop_every : 0;
        binding->processing_delay_us =
            processing_delay_us > 0 ? processing_delay_us : 0;
        send_control_ack(
            ack_socket, ack_token, operation, request_id, stage, 1, NULL);
        return;
    }
    if (strcmp(operation, "drain") == 0) {
        char ack[sizeof(binding->drain_ack)] = {0};
        double timeout = drain_timeout_ms;
        double idle = drain_idle_ms;
        json_string(line, "drain_ack", ack, sizeof(ack));
        json_double(line, "drain_timeout_ms", &timeout);
        json_double(line, "drain_idle_ms", &idle);
        binding->draining = 1;
        binding->drain_requested_ms = monotonic_ms();
        binding->drain_deadline_ms = binding->drain_requested_ms + (timeout > 0.0 ? timeout : drain_timeout_ms);
        binding->drain_idle_ms = idle > 0.0 ? idle : drain_idle_ms;
        snprintf(binding->drain_ack, sizeof(binding->drain_ack), "%s", ack);
        snprintf(binding->drain_ack_socket, sizeof(binding->drain_ack_socket), "%s", ack_socket);
        snprintf(binding->drain_ack_token, sizeof(binding->drain_ack_token), "%s", ack_token);
        binding->drain_last_packet_ms = binding->drain_requested_ms;
        return;
    }
    if (strcmp(operation, "unregister") == 0) {
        uint64_t received = binding->received;
        uint64_t forwarded = binding->forwarded;
        close_binding(binding);
        send_unregister_ack(
            ack_socket, ack_token, request_id, stage, received, forwarded);
    }
}

static void process_packets_scalar(binding_t *binding, int burst) {
    unsigned char payload[MAX_PAYLOAD];
    for (int i = 0; i < burst; i++) {
        ssize_t length = recv(binding->receiver_fd, payload, sizeof(payload), 0);
        if (length < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;
            }
            binding->dropped++;
            break;
        }
        binding->received++;
        if (binding->draining) {
            binding->drain_last_packet_ms = monotonic_ms();
        }
        if (binding->drop_every > 0 &&
            binding->received % (uint64_t)binding->drop_every == 0) {
            binding->dropped++;
            continue;
        }
        if (binding->processing_delay_us > 0) {
            struct timespec delay = {
                .tv_sec = binding->processing_delay_us / 1000000,
                .tv_nsec = (long)(binding->processing_delay_us % 1000000) * 1000L,
            };
            while (nanosleep(&delay, &delay) < 0 && errno == EINTR) {
            }
        }
        ssize_t sent = sendto(
            binding->sender_fd,
            payload,
            (size_t)length,
            MSG_DONTWAIT,
            (struct sockaddr *)&binding->next_addr,
            sizeof(binding->next_addr));
        if (sent == length) {
            binding->forwarded++;
        } else {
            binding->dropped++;
        }
    }
}

static void process_packets(binding_t *binding, int burst) {
    if (
        burst > MAX_IO_BATCH ||
        binding->drop_every > 0 ||
        binding->processing_delay_us > 0
    ) {
        process_packets_scalar(binding, burst);
        return;
    }

    static unsigned char payloads[MAX_IO_BATCH][MAX_PAYLOAD];
    struct mmsghdr receive_messages[MAX_IO_BATCH];
    struct iovec receive_iovecs[MAX_IO_BATCH];
    int batch = burst < MAX_IO_BATCH ? burst : MAX_IO_BATCH;
    memset(receive_messages, 0, sizeof(receive_messages));
    for (int i = 0; i < batch; i++) {
        receive_iovecs[i].iov_base = payloads[i];
        receive_iovecs[i].iov_len = sizeof(payloads[i]);
        receive_messages[i].msg_hdr.msg_iov = &receive_iovecs[i];
        receive_messages[i].msg_hdr.msg_iovlen = 1;
    }
    int received = recvmmsg(
        binding->receiver_fd,
        receive_messages,
        (unsigned int)batch,
        MSG_DONTWAIT,
        NULL);
    if (received < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            binding->dropped++;
        }
        return;
    }
    if (received == 0) {
        return;
    }
    binding->received += (uint64_t)received;
    if (binding->draining) {
        binding->drain_last_packet_ms = monotonic_ms();
    }

    int forwarded = 0;
    for (int i = 0; i < received; i++) {
        ssize_t sent = sendto(
            binding->sender_fd,
            payloads[i],
            receive_messages[i].msg_len,
            MSG_DONTWAIT,
            (struct sockaddr *)&binding->next_addr,
            sizeof(binding->next_addr));
        if (sent == (ssize_t)receive_messages[i].msg_len) {
            forwarded++;
        }
    }
    binding->forwarded += (uint64_t)forwarded;
    binding->dropped += (uint64_t)(received - forwarded);
}

static void check_drains(binding_t *bindings) {
    double now = monotonic_ms();
    for (int i = 0; i < MAX_BINDINGS; i++) {
        binding_t *binding = &bindings[i];
        if (!binding->used || !binding->draining) {
            continue;
        }
        int timed_out = now >= binding->drain_deadline_ms;
        int idle = now - binding->drain_last_packet_ms >= binding->drain_idle_ms;
        if (timed_out || idle) {
            binding->draining = 0;
            write_drain_ack(binding, timed_out);
            send_drain_control_ack(binding, timed_out);
        }
    }
}

static void cleanup_bindings(binding_t *bindings) {
    for (int i = 0; i < MAX_BINDINGS; i++) {
        if (bindings[i].used) {
            close_binding(&bindings[i]);
        }
    }
}

static int dscp_service_class(int dscp) {
    if (dscp >= 46) return 2;
    if (dscp >= 34) return 1;
    return 0;
}

int main(int argc, char **argv) {
    const char *fifo_path = NULL;
    const char *ready_path = NULL;
    int packet_burst = DEFAULT_PACKET_BURST;
    int q0_packet_burst = 0;
    int dscp_scheduling = 0;
    int realtime_priority = 0;
    double drain_timeout_ms = DEFAULT_DRAIN_TIMEOUT_MS;
    double drain_idle_ms = DEFAULT_DRAIN_IDLE_MS;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--command-fifo") == 0 && i + 1 < argc) fifo_path = argv[++i];
        else if (strcmp(argv[i], "--ready-file") == 0 && i + 1 < argc) ready_path = argv[++i];
        else if (strcmp(argv[i], "--max-packets-per-socket-event") == 0 && i + 1 < argc) packet_burst = atoi(argv[++i]);
        else if (strcmp(argv[i], "--q0-max-packets-per-socket-event") == 0 && i + 1 < argc) q0_packet_burst = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dscp-scheduling") == 0) dscp_scheduling = 1;
        else if (strcmp(argv[i], "--realtime-priority") == 0 && i + 1 < argc) realtime_priority = atoi(argv[++i]);
        else if (strcmp(argv[i], "--drain-timeout-ms") == 0 && i + 1 < argc) drain_timeout_ms = atof(argv[++i]);
        else if (strcmp(argv[i], "--drain-idle-ms") == 0 && i + 1 < argc) drain_idle_ms = atof(argv[++i]);
    }
    if (!fifo_path || !ready_path || packet_burst <= 0 || q0_packet_burst < 0) {
        fprintf(stderr, "usage: vnf_agent_native --command-fifo FIFO --ready-file FILE [options]\n");
        return 2;
    }
    signal(SIGTERM, stop_signal);
    signal(SIGINT, stop_signal);
    if (realtime_priority > 0) {
        struct sched_param param;
        param.sched_priority = realtime_priority;
        if (sched_setscheduler(0, SCHED_RR, &param) < 0) {
            fprintf(stderr, "native VNF agent: SCHED_RR unavailable: %s\n", strerror(errno));
        }
    }
    unlink(fifo_path);
    if (mkfifo(fifo_path, 0666) < 0 && errno != EEXIST) {
        perror("mkfifo");
        return 1;
    }
    int fifo_fd = open(fifo_path, O_RDWR | O_NONBLOCK);
    if (fifo_fd < 0) {
        perror("open command fifo");
        return 1;
    }
    static binding_t bindings[MAX_BINDINGS];
    memset(bindings, 0, sizeof(bindings));
    for (int i = 0; i < MAX_BINDINGS; i++) {
        bindings[i].receiver_fd = -1;
        bindings[i].sender_fd = -1;
    }
    char ready_json[512];
    snprintf(ready_json, sizeof(ready_json),
             "{\"backend\":\"native\",\"io_backend\":\"recvmmsg_sendto\",\"max_packets_per_socket_event\":%d,\"ready\":true,\"runtime\":\"native_vnf_agent\",\"scheduler\":\"%s\"}\n",
             packet_burst,
             dscp_scheduling ? "dscp_class_cycle" : "binding_round_robin");
    write_text_atomic(ready_path, ready_json);

    char control[MAX_JSON_LINE];
    size_t control_len = 0;
    struct pollfd pollfds[MAX_BINDINGS + 1];
    int poll_binding_indices[MAX_BINDINGS];
    while (!g_stopped) {
        int nfds = 1;
        pollfds[0].fd = fifo_fd;
        pollfds[0].events = POLLIN;
        int has_draining = 0;
        for (int i = 0; i < MAX_BINDINGS; i++) {
            if (!bindings[i].used) continue;
            if (bindings[i].draining) has_draining = 1;
            pollfds[nfds].fd = bindings[i].receiver_fd;
            pollfds[nfds].events = POLLIN;
            poll_binding_indices[nfds - 1] = i;
            nfds++;
        }
        int timeout = has_draining ? 2 : 100;
        int ready = poll(pollfds, (nfds_t)nfds, timeout);
        if (ready < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (pollfds[0].revents & POLLIN) {
            char chunk[16384];
            ssize_t count;
            while ((count = read(fifo_fd, chunk, sizeof(chunk))) > 0) {
                if (control_len + (size_t)count >= sizeof(control)) {
                    control_len = 0;
                    fprintf(stderr, "native VNF agent: control command too long\n");
                    continue;
                }
                memcpy(control + control_len, chunk, (size_t)count);
                control_len += (size_t)count;
                size_t start = 0;
                for (size_t pos = 0; pos < control_len; pos++) {
                    if (control[pos] != '\n') continue;
                    control[pos] = '\0';
                    if (control[start]) {
                        handle_command(bindings, control + start, packet_burst, q0_packet_burst, drain_timeout_ms, drain_idle_ms);
                    }
                    start = pos + 1;
                }
                if (start > 0) {
                    memmove(control, control + start, control_len - start);
                    control_len -= start;
                }
            }
        }
        int first_class = dscp_scheduling ? 2 : 0;
        for (int service_class = first_class; service_class >= 0; service_class--) {
            for (int poll_index = 1; poll_index < nfds; poll_index++) {
                binding_t *binding = &bindings[poll_binding_indices[poll_index - 1]];
                short revents = pollfds[poll_index].revents;
                if (!binding->used || !(revents & POLLIN)) continue;
                if (
                    dscp_scheduling &&
                    dscp_service_class(binding->dscp) != service_class
                ) continue;
                int burst =
                    binding->dscp >= 46 && q0_packet_burst > 0
                    ? q0_packet_burst
                    : packet_burst;
                process_packets(binding, burst);
            }
        }
        check_drains(bindings);
    }
    cleanup_bindings(bindings);
    close(fifo_fd);
    unlink(fifo_path);
    unlink(ready_path);
    return 0;
}
