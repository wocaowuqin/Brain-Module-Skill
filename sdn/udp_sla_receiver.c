#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define HEADER_BYTES 16
#define MAX_SEQUENCE_CAPACITY 50000000U

static int64_t clock_ns(clockid_t clock_id) {
    struct timespec value;
    if (clock_gettime(clock_id, &value) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (int64_t)value.tv_sec * 1000000000LL + value.tv_nsec;
}

static uint64_t be64_to_host(uint64_t value) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return ((uint64_t)ntohl((uint32_t)value) << 32) |
        ntohl((uint32_t)(value >> 32));
#else
    return value;
#endif
}

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double percentile(double *values, size_t count, double quantile) {
    if (count == 0) {
        return 0.0;
    }
    qsort(values, count, sizeof(*values), compare_double);
    size_t index = (size_t)(quantile * (double)count);
    if ((double)index < quantile * (double)count) {
        index += 1;
    }
    index = index > 0 ? index - 1 : 0;
    if (index >= count) {
        index = count - 1;
    }
    return values[index];
}

static int ensure_seen(uint8_t **seen, size_t *capacity, uint32_t required) {
    if ((size_t)required <= *capacity) {
        return 0;
    }
    if (required > MAX_SEQUENCE_CAPACITY) {
        return -1;
    }
    size_t next = *capacity ? *capacity : 1024;
    while (next < (size_t)required) {
        next *= 2;
        if (next > MAX_SEQUENCE_CAPACITY) {
            next = MAX_SEQUENCE_CAPACITY;
        }
    }
    uint8_t *expanded = realloc(*seen, next);
    if (expanded == NULL) {
        return -1;
    }
    memset(expanded + *capacity, 0, next - *capacity);
    *seen = expanded;
    *capacity = next;
    return 0;
}

static int append_delay(double **values, size_t *count, size_t *capacity, double value) {
    if (*count == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 1024;
        double *expanded = realloc(*values, next * sizeof(*expanded));
        if (expanded == NULL) {
            return -1;
        }
        *values = expanded;
        *capacity = next;
    }
    (*values)[(*count)++] = value;
    return 0;
}

static int read_expected(const char *path) {
    if (strcmp(path, "-") == 0) {
        return 0;
    }
    FILE *handle = fopen(path, "r");
    if (handle == NULL) {
        return 0;
    }
    int value = 0;
    if (fscanf(handle, "%d", &value) != 1) {
        value = 0;
    }
    fclose(handle);
    return value;
}

static void write_ready(const char *path, int64_t ready_time_ns) {
    if (strcmp(path, "-") == 0) {
        return;
    }
    FILE *handle = fopen(path, "w");
    if (handle == NULL) {
        perror("fopen ready file");
        exit(2);
    }
    fprintf(handle, "%lld", (long long)ready_time_ns);
    fclose(handle);
}

static void send_ready_ack(
    const char *socket_path,
    const char *token,
    int request_id,
    int destination_id,
    int64_t ready_monotonic_ns,
    int64_t ready_time_ns
) {
    if (strcmp(socket_path, "-") == 0 || strcmp(token, "-") == 0) {
        return;
    }
    int descriptor = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (descriptor < 0) {
        perror("socket AF_UNIX");
        exit(2);
    }
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    snprintf(address.sun_path, sizeof(address.sun_path), "%s", socket_path);
    char payload[1024];
    int length = snprintf(
        payload,
        sizeof(payload),
        "{\"accepted\":true,\"ack_token\":\"%s\","
        "\"operation\":\"receiver_ready\","
        "\"ready_monotonic_ns\":%lld,\"ready_time_ns\":%lld,"
        "\"request_id\":%d,\"destination_id\":%d}",
        token,
        (long long)ready_monotonic_ns,
        (long long)ready_time_ns,
        request_id,
        destination_id
    );
    if (
        length <= 0 || (size_t)length >= sizeof(payload) ||
        sendto(
            descriptor,
            payload,
            (size_t)length,
            0,
            (struct sockaddr *)&address,
            sizeof(address)
        ) != length
    ) {
        perror("send receiver ready ACK");
        close(descriptor);
        exit(2);
    }
    close(descriptor);
}

static void usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s GROUP PORT DURATION INTERFACE DELAY_BOUND RATIO "
        "JITTER_BOUND LOSS_BOUND GRACE EXPECTED_HINT READY_FILE STOP_NS "
        "EXPECTED_FILE RCVBUF ACK_SOCKET ACK_TOKEN REQUEST_ID DEST_ID\n",
        program
    );
}

int main(int argc, char **argv) {
    if (argc != 19) {
        usage(argv[0]);
        return 2;
    }
    const char *group = argv[1];
    int port = atoi(argv[2]);
    double duration = strtod(argv[3], NULL);
    const char *interface_ip = argv[4];
    double delay_bound_ms = strtod(argv[5], NULL);
    double compliance_ratio = strtod(argv[6], NULL);
    double jitter_bound_ms = strtod(argv[7], NULL);
    double loss_bound = strtod(argv[8], NULL);
    double grace_seconds = strtod(argv[9], NULL);
    int expected_hint = atoi(argv[10]);
    const char *ready_file = argv[11];
    int64_t stop_time_ns = strtoll(argv[12], NULL, 10);
    const char *expected_file = argv[13];
    int receive_buffer_bytes = atoi(argv[14]);
    const char *ack_socket = argv[15];
    const char *ack_token = argv[16];
    int request_id = atoi(argv[17]);
    int destination_id = atoi(argv[18]);
    if (
        port <= 0 || port > 65535 || duration <= 0.0 || delay_bound_ms <= 0.0 ||
        compliance_ratio <= 0.0 || compliance_ratio > 1.0 || loss_bound < 0.0 ||
        loss_bound > 1.0 || grace_seconds < 0.0 || expected_hint < 0 ||
        receive_buffer_bytes <= 0
    ) {
        usage(argv[0]);
        return 2;
    }

    int descriptor = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (descriptor < 0) {
        perror("socket");
        return 2;
    }
    int enabled = 1;
    if (
        setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled)) != 0 ||
        setsockopt(
            descriptor,
            SOL_SOCKET,
            SO_RCVBUF,
            &receive_buffer_bytes,
            sizeof(receive_buffer_bytes)
        ) != 0 ||
        setsockopt(descriptor, SOL_SOCKET, SO_TIMESTAMPNS, &enabled, sizeof(enabled)) != 0
    ) {
        perror("setsockopt");
        close(descriptor);
        return 2;
    }
    int actual_receive_buffer_bytes = 0;
    socklen_t option_length = sizeof(actual_receive_buffer_bytes);
    getsockopt(
        descriptor,
        SOL_SOCKET,
        SO_RCVBUF,
        &actual_receive_buffer_bytes,
        &option_length
    );
    struct timeval timeout = {.tv_sec = 0, .tv_usec = 200000};
    setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    struct sockaddr_in bind_address;
    memset(&bind_address, 0, sizeof(bind_address));
    bind_address.sin_family = AF_INET;
    bind_address.sin_addr.s_addr = htonl(INADDR_ANY);
    bind_address.sin_port = htons((uint16_t)port);
    if (
        bind(
            descriptor,
            (struct sockaddr *)&bind_address,
            sizeof(bind_address)
        ) != 0
    ) {
        perror("bind");
        close(descriptor);
        return 2;
    }
    struct ip_mreq membership;
    if (
        inet_pton(AF_INET, group, &membership.imr_multiaddr) != 1 ||
        inet_pton(AF_INET, interface_ip, &membership.imr_interface) != 1 ||
        setsockopt(
            descriptor,
            IPPROTO_IP,
            IP_ADD_MEMBERSHIP,
            &membership,
            sizeof(membership)
        ) != 0
    ) {
        perror("IP_ADD_MEMBERSHIP");
        close(descriptor);
        return 2;
    }

    int64_t setup_started_ns = clock_ns(CLOCK_MONOTONIC);
    int64_t ready_monotonic_ns = clock_ns(CLOCK_MONOTONIC);
    int64_t ready_time_ns = clock_ns(CLOCK_REALTIME);
    double setup_seconds = (ready_monotonic_ns - setup_started_ns) / 1e9;
    write_ready(ready_file, ready_time_ns);
    send_ready_ack(
        ack_socket,
        ack_token,
        request_id,
        destination_id,
        ready_monotonic_ns,
        ready_time_ns
    );

    double remaining_seconds = duration;
    if (stop_time_ns > 0) {
        remaining_seconds = (stop_time_ns - clock_ns(CLOCK_REALTIME)) / 1e9;
        if (remaining_seconds < 0.0) {
            remaining_seconds = 0.0;
        }
    }
    int64_t deadline_ns = ready_monotonic_ns +
        (int64_t)((remaining_seconds + grace_seconds) * 1e9);
    int64_t realtime_monotonic_offset =
        clock_ns(CLOCK_REALTIME) - clock_ns(CLOCK_MONOTONIC);

    uint8_t *seen = NULL;
    size_t seen_capacity = 0;
    double *delays = NULL;
    size_t delay_count = 0;
    size_t delay_capacity = 0;
    uint32_t expected_packets = 0;
    double jitter_ms = 0.0;
    double previous_transit_ms = 0.0;
    int have_previous = 0;
    unsigned char payload[65535];
    while (clock_ns(CLOCK_MONOTONIC) < deadline_ns) {
        struct iovec vector = {.iov_base = payload, .iov_len = sizeof(payload)};
        char control[CMSG_SPACE(sizeof(struct timespec))];
        struct msghdr message;
        memset(&message, 0, sizeof(message));
        message.msg_iov = &vector;
        message.msg_iovlen = 1;
        message.msg_control = control;
        message.msg_controllen = sizeof(control);
        ssize_t length = recvmsg(descriptor, &message, 0);
        if (length < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                continue;
            }
            perror("recvmsg");
            free(seen);
            free(delays);
            close(descriptor);
            return 2;
        }
        if (length < HEADER_BYTES) {
            continue;
        }
        uint32_t sequence_be;
        uint32_t total_be;
        uint64_t sent_be;
        memcpy(&sequence_be, payload, sizeof(sequence_be));
        memcpy(&total_be, payload + 4, sizeof(total_be));
        memcpy(&sent_be, payload + 8, sizeof(sent_be));
        uint32_t sequence = ntohl(sequence_be);
        uint32_t total = ntohl(total_be);
        uint64_t sent_ns = be64_to_host(sent_be);
        if (ensure_seen(&seen, &seen_capacity, sequence + 1) != 0) {
            fprintf(stderr, "sequence range is too large\n");
            free(seen);
            free(delays);
            close(descriptor);
            return 2;
        }
        if (seen[sequence]) {
            continue;
        }
        seen[sequence] = 1;
        if (total > expected_packets) {
            expected_packets = total;
        }
        int64_t arrival_realtime_ns = 0;
        for (
            struct cmsghdr *header = CMSG_FIRSTHDR(&message);
            header != NULL;
            header = CMSG_NXTHDR(&message, header)
        ) {
            if (
                header->cmsg_level == SOL_SOCKET &&
                header->cmsg_type == SCM_TIMESTAMPNS
            ) {
                struct timespec *timestamp = (struct timespec *)CMSG_DATA(header);
                arrival_realtime_ns =
                    (int64_t)timestamp->tv_sec * 1000000000LL + timestamp->tv_nsec;
                break;
            }
        }
        int64_t arrival_monotonic_ns = arrival_realtime_ns > 0
            ? arrival_realtime_ns - realtime_monotonic_offset
            : clock_ns(CLOCK_MONOTONIC);
        double transit_ms = arrival_monotonic_ns > (int64_t)sent_ns
            ? (arrival_monotonic_ns - (int64_t)sent_ns) / 1e6
            : 0.0;
        if (append_delay(
                &delays,
                &delay_count,
                &delay_capacity,
                transit_ms
            ) != 0) {
            perror("realloc delays");
            free(seen);
            free(delays);
            close(descriptor);
            return 2;
        }
        if (have_previous) {
            double difference = transit_ms - previous_transit_ms;
            if (difference < 0.0) {
                difference = -difference;
            }
            jitter_ms += (difference - jitter_ms) / 16.0;
        }
        previous_transit_ms = transit_ms;
        have_previous = 1;
    }
    close(descriptor);

    int expected_from_file = read_expected(expected_file);
    int sender_failed = expected_from_file < 0;
    uint32_t received = (uint32_t)delay_count;
    uint32_t expected = expected_from_file > 0
        ? (uint32_t)expected_from_file
        : expected_packets;
    if ((uint32_t)expected_hint > expected) {
        expected = (uint32_t)expected_hint;
    }
    if (received > expected) {
        expected = received;
    }
    uint32_t lost = expected > received ? expected - received : 0;
    uint32_t delay_violations = 0;
    double delay_total = 0.0;
    double delay_max = 0.0;
    for (size_t index = 0; index < delay_count; ++index) {
        delay_total += delays[index];
        if (delays[index] > delay_max) {
            delay_max = delays[index];
        }
        if (delays[index] > delay_bound_ms) {
            delay_violations += 1;
        }
    }
    double delay_violation_rate =
        (double)delay_violations / (double)(received ? received : 1);
    double loss_rate = sender_failed
        ? 1.0
        : (double)lost / (double)(expected ? expected : 1);
    int delay_sla = received > 0 &&
        delay_violation_rate <= 1.0 - compliance_ratio + 1e-12;
    int jitter_sla = jitter_bound_ms < 0.0 ||
        (received > 0 && jitter_ms <= jitter_bound_ms);
    int loss_sla = !sender_failed && loss_rate <= loss_bound + 1e-12;

    double p50 = percentile(delays, delay_count, 0.50);
    double p95 = percentile(delays, delay_count, 0.95);
    double p99 = percentile(delays, delay_count, 0.99);
    int64_t first_missing = -1;
    int64_t last_missing = -1;
    uint32_t represented_missing = 0;
    uint32_t total_missing = 0;
    char ranges[2048];
    size_t ranges_length = 0;
    ranges[ranges_length++] = '[';
    int range_count = 0;
    for (uint32_t sequence = 0; sequence < expected;) {
        int present = sequence < seen_capacity && seen[sequence];
        if (present) {
            sequence += 1;
            continue;
        }
        uint32_t start = sequence;
        while (
            sequence < expected &&
            !(sequence < seen_capacity && seen[sequence])
        ) {
            sequence += 1;
        }
        uint32_t end = sequence - 1;
        uint32_t length = end - start + 1;
        total_missing += length;
        if (first_missing < 0) {
            first_missing = start;
        }
        last_missing = end;
        if (range_count < 16) {
            int written = snprintf(
                ranges + ranges_length,
                sizeof(ranges) - ranges_length,
                "%s[%u,%u]",
                range_count ? "," : "",
                start,
                end
            );
            if (written > 0) {
                ranges_length += (size_t)written;
            }
            represented_missing += length;
            range_count += 1;
        }
    }
    ranges[ranges_length++] = ']';
    ranges[ranges_length] = '\0';
    uint32_t out_of_range = 0;
    for (size_t sequence = expected; sequence < seen_capacity; ++sequence) {
        out_of_range += seen[sequence] != 0;
    }

    printf(
        "{\"mode\":\"receiver\",\"group\":\"%s\",\"port\":%d,"
        "\"interface_ip\":\"%s\",\"expected_packets\":%u,"
        "\"received_packets\":%u,\"lost_packets\":%u,"
        "\"packet_loss_rate\":%.12f,\"first_missing_sequence\":",
        group,
        port,
        interface_ip,
        expected,
        received,
        lost,
        loss_rate
    );
    if (first_missing < 0) printf("null"); else printf("%lld", (long long)first_missing);
    printf(",\"last_missing_sequence\":");
    if (last_missing < 0) printf("null"); else printf("%lld", (long long)last_missing);
    printf(
        ",\"missing_sequence_ranges\":%s,\"missing_sequences_omitted\":%u,"
        "\"out_of_range_sequences\":%u,\"mean_delay_ms\":",
        ranges,
        total_missing - represented_missing,
        out_of_range
    );
    if (delay_count) printf("%.9f", delay_total / (double)delay_count); else printf("null");
    printf(",\"p50_delay_ms\":");
    if (delay_count) printf("%.9f", p50); else printf("null");
    printf(",\"p95_delay_ms\":");
    if (delay_count) printf("%.9f", p95); else printf("null");
    printf(",\"p99_delay_ms\":");
    if (delay_count) printf("%.9f", p99); else printf("null");
    printf(",\"max_delay_ms\":");
    if (delay_count) printf("%.9f", delay_max); else printf("null");
    printf(",\"jitter_ms\":");
    if (delay_count) printf("%.9f", jitter_ms); else printf("null");
    printf(
        ",\"delay_bound_ms\":%.9f,\"delay_compliance_ratio\":%.9f,"
        "\"delay_violations\":%u,\"delay_violation_rate\":%.12f,"
        "\"jitter_bound_ms\":",
        delay_bound_ms,
        compliance_ratio,
        delay_violations,
        delay_violation_rate
    );
    if (jitter_bound_ms >= 0.0) printf("%.9f", jitter_bound_ms); else printf("null");
    printf(
        ",\"packet_loss_bound\":%.9f,\"receiver_setup_seconds\":%.9f,"
        "\"receive_buffer_bytes\":%d,\"expected_packets_source\":\"%s\","
        "\"measurement_status\":\"%s\",\"delay_sla_met\":%s,"
        "\"jitter_sla_met\":%s,\"loss_sla_met\":%s,\"sla_met\":%s,"
        "\"receiver_backend\":\"native_c_kernel_timestamp\"}\n",
        loss_bound,
        setup_seconds,
        actual_receive_buffer_bytes,
        sender_failed ? "sender_failure" :
            expected_from_file > 0 ? "sender_file" :
            expected_packets > 0 ? "packet_header" :
            expected_hint > 0 ? "hint" : "none",
        sender_failed ? "sender_failed" : "completed",
        delay_sla ? "true" : "false",
        jitter_sla ? "true" : "false",
        loss_sla ? "true" : "false",
        delay_sla && jitter_sla && loss_sla ? "true" : "false"
    );
    free(seen);
    free(delays);
    return 0;
}
