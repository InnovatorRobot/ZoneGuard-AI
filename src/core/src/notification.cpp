#include "core/notification.h"

#include <iostream>
#include <utility>

namespace ZoneGuardAI
{
namespace Core
{
NotificationClient::NotificationClient()
{
    sinks_.push_back(consoleSink());
}

void NotificationClient::addSink(Sink sink)
{
    if (!sink)
    {
        return;
    }
    std::lock_guard<std::mutex> lock{mutex_};
    sinks_.push_back(std::move(sink));
}

void NotificationClient::setCooldownMs(std::int64_t cooldownMs)
{
    std::lock_guard<std::mutex> lock{mutex_};
    cooldown_ms_ = cooldownMs;
}

bool NotificationClient::notify(Alert const& alert)
{
    std::vector<Sink> sinksCopy{};
    {
        std::lock_guard<std::mutex> lock{mutex_};

        std::string const key{std::to_string(alert.trackId) + '|' + alert.zoneName};
        auto const it{last_dispatch_ms_.find(key)};
        if (it != last_dispatch_ms_.end() && (alert.timestampMs - it->second) < cooldown_ms_)
        {
            return false;
        }
        last_dispatch_ms_[key] = alert.timestampMs;
        sinksCopy              = sinks_;
    }

    for (Sink const& sink : sinksCopy)
    {
        sink(alert);
    }
    return true;
}

NotificationClient::Sink NotificationClient::consoleSink()
{
    return [](Alert const& alert) {
        std::cerr << "[ALERT] track " << alert.trackId << " in zone '" << alert.zoneName
                  << "': " << alert.action << " (" << alert.confidence * 100.0F << "%) @ "
                  << alert.timestampMs << "ms" << std::endl;
    };
}

}  // namespace Core
}  // namespace ZoneGuardAI
