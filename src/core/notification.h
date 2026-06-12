#ifndef ZoneGuardAI_CORE_NOTIFICATION_H_
#define ZoneGuardAI_CORE_NOTIFICATION_H_

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ZoneGuardAI
{
namespace Core
{
/**
 * A single alert event raised when a fall is detected inside a monitoring zone.
 *
 * `timestampMs` is wall-clock Unix time in milliseconds so downstream sinks
 * (logs, UI panel, webhooks) can present a consistent time.
 */
struct Alert
{
    std::int32_t trackId{-1};
    std::string zoneName{};
    std::string action{};
    float confidence{0.0F};
    std::int64_t timestampMs{0};
};

/**
 * Dispatches alerts to one or more sinks, with per-source debouncing so a
 * sustained fall does not flood notifications every frame.
 *
 * The client is Qt-free and thread-safe: the pipeline calls `notify` from its
 * worker thread, and sinks (a console logger by default, plus any UI/webhook
 * handlers) receive de-duplicated events. Alerts for the same track and zone
 * within the cooldown window are suppressed.
 */
class NotificationClient
{
 public:
    using Sink = std::function<void(Alert const&)>;

    NotificationClient();

    /** Register a sink invoked for every alert that passes debouncing. */
    void addSink(Sink sink);

    /** Cooldown between repeated alerts for the same track+zone (ms). */
    void setCooldownMs(std::int64_t cooldownMs);

    /**
     * Submit an alert. Returns true when it was dispatched, false when it was
     * suppressed by the cooldown. Thread-safe.
     */
    bool notify(Alert const& alert);

    /** A sink that writes a human-readable line to stderr. */
    static Sink consoleSink();

 private:
    mutable std::mutex mutex_;
    std::vector<Sink> sinks_{};
    std::map<std::string, std::int64_t> last_dispatch_ms_{};
    std::int64_t cooldown_ms_{5000};
};

}  // namespace Core
}  // namespace ZoneGuardAI
#endif  // ZoneGuardAI_CORE_NOTIFICATION_H_
