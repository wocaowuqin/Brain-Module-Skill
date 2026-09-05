#define _POSIX_C_SOURCE 200809L

#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define HEADER_BYTES 16
#define LATE_WAKEUP_NS 50000LL
#define SPIN_WINDOW_NS 100000LL
/* The scheduled sender stop precedes request teardown by 500 ms by default.
 * Keep a bounded grace here so a final absolute-time slot is not discarded
 * solely because scheduler wake-up crosses the artificial sender boundary. */
#define STOP_GRACE_NS 50000000LL

static int64_t clock_ns(clockid_t clock_id) {
    struct timespec value;
    if (clock_gettime(clock_id, &value) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (int64_t)value.tv_sec * 1000000000LL + value.tv_nsec;
}

static uint64_t host_to_be64(uint64_t value) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return ((uint64_t)htonl((uint32_t)value) << 32) |
        htonl((uint32_t)(value >> 32));
#else
    return value;
#endif
}

static void sleep_until_ns(int64_t target_ns) {
    int64_t sleep_target = target_ns - SPIN_WINDOW_NS;
    if (clock_ns(CLOCK_MONOTONIC) < sleep_target) {
        struct timespec value = {
            .tv_sec = sleep_target / 1000000000LL,
            .tv_nsec = sleep_target % 1000000000LL,
        };
        int status;
        do {
            status = clock_nanosleep(
                CLOCK_MONOTONIC, TIMER_ABSTIME, &value, NULL
            );
        } while (status == EINTR);
        if (status != 0) {
            errno = status;
            perror("clock_nanosleep");
            exit(2);
        }
    }
    while (clock_ns(CLOCK_MONOTONIC) < target_ns) {
        __asm__ __volatile__("pause");
    }
}

static void usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s DEST PORT DURATION PPS PAYLOAD DSCP TTL STOP_TIME_NS EXPECTED_FILE\n",
        program
    );
}

int main(int argc, char **argv) {
    if (argc != 10) {
        usage(argv[0]);
        return 2;
    }
    const char *destination = argv[1];
    int port = atoi(argv[2]);
    double duration = strtod(argv[3], NULL);
    double pps = strtod(argv[4], NULL);
    int payload_bytes = atoi(argv[5]);
    int dscp = atoi(argv[6]);
    int ttl = atoi(argv[7]);
    int64_t stop_time_ns = strtoll(argv[8], NULL, 10);
    const char *expected_file = argv[9];
    if (
        port <= 0 || port > 65535 || duration <= 0.0 || pps <= 0.0 ||
        payload_bytes < HEADER_BYTES || dscp < 0 || dscp > 63 ||
        ttl <= 0 || ttl > 255
    ) {
        usage(argv[0]);
        return 2;
    }

    int64_t interval_ns = (int64_t)llround(1000000000.0 / pps);
    if (interval_ns < 1) {
        interval_ns = 1;
    }

    int descriptor = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (descriptor < 0) {
        perror("socket");
        return 2;
    }
    int tos = dscp << 2;
    if (setsockopt(descriptor, IPPROTO_IP, IP_TOS, &tos, sizeof(tos)) != 0) {
        perror("setsockopt IP_TOS");
        close(descriptor);
        return 2;
    }
    unsigned char multicast_ttl = (unsigned char)ttl;
    if (
        setsockopt(
            descriptor,
            IPPROTO_IP,
            IP_MULTICAST_TTL,
            &multicast_ttl,
            sizeof(multicast_ttl)
        ) != 0
    ) {
        perror("setsockopt IP_MULTICAST_TTL");
        close(descriptor);
        return 2;
    }

    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, destination, &address.sin_addr) != 1) {
        fprintf(stderr, "invalid IPv4 destination: %s\n", destination);
        close(descriptor);
        return 2;
    }

    unsigned char *payload = calloc((size_t)payload_bytes, 1);
    if (payload == NULL) {
        perror("calloc");
        close(descriptor);
        return 2;
    }

    int64_t started_ns = clock_ns(CLOCK_MONOTONIC);
    int64_t stop_monotonic_ns = 0;
    if (stop_time_ns > 0) {
        int64_t remaining_ns = stop_time_ns - clock_ns(CLOCK_REALTIME);
        double remaining = remaining_ns / 1e9;
        if (remaining < duration) {
            duration = remaining;
        }
        stop_monotonic_ns = started_ns + remaining_ns;
    }
    if (duration <= 0.0) {
        fprintf(stderr, "sender deadline already expired\n");
        free(payload);
        close(descriptor);
        return 2;
    }
    uint32_t planned = (uint32_t)fmax(1.0, ceil(duration * pps));
    uint32_t sent = 0;
    uint64_t late_wakeups = 0;
    uint64_t catch_up_packets = 0;
    uint64_t catch_up_burst = 0;
    uint64_t max_catch_up_burst = 0;
    int64_t max_lateness_ns = 0;
    long double total_lateness_ns = 0.0;

    for (uint32_t sequence = 0; sequence < planned; ++sequence) {
        if (
            stop_monotonic_ns > 0 &&
            clock_ns(CLOCK_MONOTONIC) >= stop_monotonic_ns + STOP_GRACE_NS
        ) {
            break;
        }
        int64_t target_ns = started_ns + (int64_t)sequence * interval_ns;
        sleep_until_ns(target_ns);
        int64_t woke_ns = clock_ns(CLOCK_MONOTONIC);
        int64_t lateness_ns = woke_ns > target_ns ? woke_ns - target_ns : 0;
        if (
            stop_monotonic_ns > 0 &&
            clock_ns(CLOCK_MONOTONIC) >= stop_monotonic_ns + STOP_GRACE_NS
        ) {
            break;
        }
        if (lateness_ns >= interval_ns) {
            catch_up_packets += 1;
            catch_up_burst += 1;
            if (catch_up_burst > max_catch_up_burst) {
                max_catch_up_burst = catch_up_burst;
            }
        } else {
            catch_up_burst = 0;
        }
        if (lateness_ns > LATE_WAKEUP_NS) {
            late_wakeups += 1;
        }
        if (lateness_ns > max_lateness_ns) {
            max_lateness_ns = lateness_ns;
        }
        total_lateness_ns += lateness_ns;

        uint32_t sequence_be = htonl(sent);
        uint32_t planned_be = htonl(planned);
        uint64_t timestamp_be = host_to_be64(
            (uint64_t)clock_ns(CLOCK_MONOTONIC)
        );
        memcpy(payload, &sequence_be, sizeof(sequence_be));
        memcpy(payload + 4, &planned_be, sizeof(planned_be));
        memcpy(payload + 8, &timestamp_be, sizeof(timestamp_be));
        ssize_t written = sendto(
            descriptor,
            payload,
            (size_t)payload_bytes,
            0,
            (struct sockaddr *)&address,
            sizeof(address)
        );
        if (written != payload_bytes) {
            perror("sendto");
            free(payload);
            close(descriptor);
            return 2;
        }
        sent += 1;
    }

    double elapsed = (clock_ns(CLOCK_MONOTONIC) - started_ns) / 1e9;
    double mean_lateness_us = sent > 0
        ? (double)(total_lateness_ns / sent / 1000.0)
        : 0.0;
    FILE *expected = fopen(expected_file, "w");
    if (expected == NULL) {
        perror("fopen expected file");
        free(payload);
        close(descriptor);
        return 2;
    }
    fprintf(expected, "%u", planned);
    if (fclose(expected) != 0) {
        perror("fclose expected file");
        free(payload);
        close(descriptor);
        return 2;
    }
    printf(
        "{\"mode\":\"sender\",\"destination\":\"%s\","
        "\"port\":%d,\"dscp\":%d,\"payload_bytes\":%d,"
        "\"packets_per_second\":%.9f,\"planned_packets\":%u,"
        "\"sent_packets\":%u,\"deadline_limited_packets\":%u,"
        "\"pacing_resets\":0,\"skipped_pacing_slots\":0,"
        "\"max_catch_up_packets\":-1,\"catch_up_packets\":%llu,"
        "\"max_catch_up_burst\":%llu,\"late_wakeups_over_50us\":%llu,"
        "\"max_pacing_lateness_us\":%.3f,"
        "\"mean_pacing_lateness_us\":%.3f,"
        "\"pacing_wait_backend\":\"clock_nanosleep_spin\","
        "\"spin_window_us\":100.0," 
        "\"sender_backend\":\"native_c\",\"stop_grace_ms\":50.0,"
        "\"elapsed_seconds\":%.9f}\n",
        destination,
        port,
        dscp,
        payload_bytes,
        pps,
        planned,
        sent,
        planned - sent,
        (unsigned long long)catch_up_packets,
        (unsigned long long)max_catch_up_burst,
        (unsigned long long)late_wakeups,
        max_lateness_ns / 1000.0,
        mean_lateness_us,
        elapsed
    );

    free(payload);
    close(descriptor);
    return 0;
}
