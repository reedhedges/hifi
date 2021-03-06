//
//  EntityServer.cpp
//  assignment-client/src/entities
//
//  Created by Brad Hefta-Gaub on 4/29/14
//  Copyright 2014 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//

#include <QtCore/QEventLoop>
#include <QTimer>
#include <EntityTree.h>
#include <SimpleEntitySimulation.h>
#include <ResourceCache.h>
#include <ScriptCache.h>

#include "EntityServer.h"
#include "EntityServerConsts.h"
#include "EntityNodeData.h"
#include "AssignmentParentFinder.h"

const char* MODEL_SERVER_NAME = "Entity";
const char* MODEL_SERVER_LOGGING_TARGET_NAME = "entity-server";
const char* LOCAL_MODELS_PERSIST_FILE = "resources/models.svo";

EntityServer::EntityServer(ReceivedMessage& message) :
    OctreeServer(message),
    _entitySimulation(NULL)
{
    ResourceManager::init();
    DependencyManager::set<ResourceCacheSharedItems>();
    DependencyManager::set<ScriptCache>();

    auto& packetReceiver = DependencyManager::get<NodeList>()->getPacketReceiver();
    packetReceiver.registerListenerForTypes({ PacketType::EntityAdd, PacketType::EntityEdit, PacketType::EntityErase, PacketType::EntityPhysics },
                                            this, "handleEntityPacket");
}

EntityServer::~EntityServer() {
    if (_pruneDeletedEntitiesTimer) {
        _pruneDeletedEntitiesTimer->stop();
        _pruneDeletedEntitiesTimer->deleteLater();
    }

    EntityTreePointer tree = std::static_pointer_cast<EntityTree>(_tree);
    tree->removeNewlyCreatedHook(this);
}

void EntityServer::handleEntityPacket(QSharedPointer<ReceivedMessage> message, SharedNodePointer senderNode) {
    if (_octreeInboundPacketProcessor) {
        _octreeInboundPacketProcessor->queueReceivedPacket(message, senderNode);
    }
}

std::unique_ptr<OctreeQueryNode> EntityServer::createOctreeQueryNode() {
    return std::unique_ptr<OctreeQueryNode> { new EntityNodeData() };
}

OctreePointer EntityServer::createTree() {
    EntityTreePointer tree = EntityTreePointer(new EntityTree(true));
    tree->createRootElement();
    tree->addNewlyCreatedHook(this);
    if (!_entitySimulation) {
        SimpleEntitySimulationPointer simpleSimulation { new SimpleEntitySimulation() };
        simpleSimulation->setEntityTree(tree);
        tree->setSimulation(simpleSimulation);
        _entitySimulation = simpleSimulation;
    }

    DependencyManager::registerInheritance<SpatialParentFinder, AssignmentParentFinder>();
    DependencyManager::set<AssignmentParentFinder>(tree);

    return tree;
}

void EntityServer::beforeRun() {
    _pruneDeletedEntitiesTimer = new QTimer();
    connect(_pruneDeletedEntitiesTimer, SIGNAL(timeout()), this, SLOT(pruneDeletedEntities()));
    const int PRUNE_DELETED_MODELS_INTERVAL_MSECS = 1 * 1000; // once every second
    _pruneDeletedEntitiesTimer->start(PRUNE_DELETED_MODELS_INTERVAL_MSECS);
}

void EntityServer::entityCreated(const EntityItem& newEntity, const SharedNodePointer& senderNode) {
}


// EntityServer will use the "special packets" to send list of recently deleted entities
bool EntityServer::hasSpecialPacketsToSend(const SharedNodePointer& node) {
    bool shouldSendDeletedEntities = false;

    // check to see if any new entities have been added since we last sent to this node...
    EntityNodeData* nodeData = static_cast<EntityNodeData*>(node->getLinkedData());
    if (nodeData) {
        quint64 deletedEntitiesSentAt = nodeData->getLastDeletedEntitiesSentAt();
        EntityTreePointer tree = std::static_pointer_cast<EntityTree>(_tree);
        shouldSendDeletedEntities = tree->hasEntitiesDeletedSince(deletedEntitiesSentAt);

        #ifdef EXTRA_ERASE_DEBUGGING
            if (shouldSendDeletedEntities) {
                int elapsed = usecTimestampNow() - deletedEntitiesSentAt;
                qDebug() << "shouldSendDeletedEntities to node:" << node->getUUID() << "deletedEntitiesSentAt:" << deletedEntitiesSentAt << "elapsed:" << elapsed;
            }
        #endif
    }

    return shouldSendDeletedEntities;
}

// FIXME - most of the old code for this was encapsulated in EntityTree, I liked that design from a data
// hiding and object oriented perspective. But that didn't really allow us to handle the case of lots
// of entities being deleted at the same time. I'd like to look to move this back into EntityTree but
// for now this works and addresses the bug.
int EntityServer::sendSpecialPackets(const SharedNodePointer& node, OctreeQueryNode* queryNode, int& packetsSent) {
    int totalBytes = 0;

    EntityNodeData* nodeData = static_cast<EntityNodeData*>(node->getLinkedData());
    if (nodeData) {

        quint64 deletedEntitiesSentAt = nodeData->getLastDeletedEntitiesSentAt();
        quint64 considerEntitiesSince = EntityTree::getAdjustedConsiderSince(deletedEntitiesSentAt);

        quint64 deletePacketSentAt = usecTimestampNow();
        EntityTreePointer tree = std::static_pointer_cast<EntityTree>(_tree);
        auto recentlyDeleted = tree->getRecentlyDeletedEntityIDs();

        packetsSent = 0;

        // create a new special packet
        std::unique_ptr<NLPacket> deletesPacket = NLPacket::create(PacketType::EntityErase);

        // pack in flags
        OCTREE_PACKET_FLAGS flags = 0;
        deletesPacket->writePrimitive(flags);

        // pack in sequence number
        auto sequenceNumber = queryNode->getSequenceNumber();
        deletesPacket->writePrimitive(sequenceNumber);

        // pack in timestamp
        OCTREE_PACKET_SENT_TIME now = usecTimestampNow();
        deletesPacket->writePrimitive(now);

        // figure out where we are now and pack a temporary number of IDs
        uint16_t numberOfIDs = 0;
        qint64 numberOfIDsPos = deletesPacket->pos();
        deletesPacket->writePrimitive(numberOfIDs);

        // we keep a multi map of entity IDs to timestamps, we only want to include the entity IDs that have been
        // deleted since we last sent to this node
        auto it = recentlyDeleted.constBegin();
        while (it != recentlyDeleted.constEnd()) {

            // if the timestamp is more recent then out last sent time, include it
            if (it.key() > considerEntitiesSince) {

                // get all the IDs for this timestamp
                const auto& entityIDsFromTime = recentlyDeleted.values(it.key());

                for (const auto& entityID : entityIDsFromTime) {

                    // check to make sure we have room for one more ID, if we don't have more
                    // room, then send out this packet and create another one
                    if (NUM_BYTES_RFC4122_UUID > deletesPacket->bytesAvailableForWrite()) {

                        // replace the count for the number of included IDs
                        deletesPacket->seek(numberOfIDsPos);
                        deletesPacket->writePrimitive(numberOfIDs);

                        // Send the current packet
                        queryNode->packetSent(*deletesPacket);
                        auto thisPacketSize = deletesPacket->getDataSize();
                        totalBytes += thisPacketSize;
                        packetsSent++;
                        DependencyManager::get<NodeList>()->sendPacket(std::move(deletesPacket), *node);

                        #ifdef EXTRA_ERASE_DEBUGGING
                            qDebug() << "EntityServer::sendSpecialPackets() sending packet packetsSent[" << packetsSent << "] size:" << thisPacketSize;
                        #endif


                        // create another packet
                        deletesPacket = NLPacket::create(PacketType::EntityErase);

                        // pack in flags
                        deletesPacket->writePrimitive(flags);

                        // pack in sequence number
                        sequenceNumber = queryNode->getSequenceNumber();
                        deletesPacket->writePrimitive(sequenceNumber);

                        // pack in timestamp
                        deletesPacket->writePrimitive(now);

                        // figure out where we are now and pack a temporary number of IDs
                        numberOfIDs = 0;
                        numberOfIDsPos = deletesPacket->pos();
                        deletesPacket->writePrimitive(numberOfIDs);
                    }

                    // FIXME - we still seem to see cases where incorrect EntityIDs get sent from the server
                    // to the client. These were causing "lost" entities like flashlights and laser pointers
                    // now that we keep around some additional history of the erased entities and resend that
                    // history for a longer time window, these entities are not "lost". But we haven't yet
                    // found/fixed the underlying issue that caused bad UUIDs to be sent to some users.
                    deletesPacket->write(entityID.toRfc4122());
                    ++numberOfIDs;

                    #ifdef EXTRA_ERASE_DEBUGGING
                        qDebug() << "EntityTree::encodeEntitiesDeletedSince() including:" << entityID;
                    #endif
                } // end for (ids)

            } // end if (it.val > sinceLast)


            ++it;
        } // end while

        // replace the count for the number of included IDs
        deletesPacket->seek(numberOfIDsPos);
        deletesPacket->writePrimitive(numberOfIDs);

        // Send the current packet
        queryNode->packetSent(*deletesPacket);
        auto thisPacketSize = deletesPacket->getDataSize();
        totalBytes += thisPacketSize;
        packetsSent++;
        DependencyManager::get<NodeList>()->sendPacket(std::move(deletesPacket), *node);
        #ifdef EXTRA_ERASE_DEBUGGING
            qDebug() << "EntityServer::sendSpecialPackets() sending packet packetsSent[" << packetsSent << "] size:" << thisPacketSize;
        #endif

        nodeData->setLastDeletedEntitiesSentAt(deletePacketSentAt);
    }

    #ifdef EXTRA_ERASE_DEBUGGING
        if (packetsSent > 0) {
            qDebug() << "EntityServer::sendSpecialPackets() sent " << packetsSent << "special packets of " 
                        << totalBytes << " total bytes to node:" << node->getUUID();
        }
    #endif

    // TODO: caller is expecting a packetLength, what if we send more than one packet??
    return totalBytes;
}


void EntityServer::pruneDeletedEntities() {
    EntityTreePointer tree = std::static_pointer_cast<EntityTree>(_tree);
    if (tree->hasAnyDeletedEntities()) {

        quint64 earliestLastDeletedEntitiesSent = usecTimestampNow() + 1; // in the future
        DependencyManager::get<NodeList>()->eachNode([&earliestLastDeletedEntitiesSent](const SharedNodePointer& node) {
            if (node->getLinkedData()) {
                EntityNodeData* nodeData = static_cast<EntityNodeData*>(node->getLinkedData());
                quint64 nodeLastDeletedEntitiesSentAt = nodeData->getLastDeletedEntitiesSentAt();
                if (nodeLastDeletedEntitiesSentAt < earliestLastDeletedEntitiesSent) {
                    earliestLastDeletedEntitiesSent = nodeLastDeletedEntitiesSentAt;
                }
            }
        });
        tree->forgetEntitiesDeletedBefore(earliestLastDeletedEntitiesSent);
    }
}

void EntityServer::readAdditionalConfiguration(const QJsonObject& settingsSectionObject) {
    bool wantEditLogging = false;
    readOptionBool(QString("wantEditLogging"), settingsSectionObject, wantEditLogging);
    qDebug("wantEditLogging=%s", debug::valueOf(wantEditLogging));

    bool wantTerseEditLogging = false;
    readOptionBool(QString("wantTerseEditLogging"), settingsSectionObject, wantTerseEditLogging);
    qDebug("wantTerseEditLogging=%s", debug::valueOf(wantTerseEditLogging));

    EntityTreePointer tree = std::static_pointer_cast<EntityTree>(_tree);

    int maxTmpEntityLifetime;
    if (readOptionInt("maxTmpLifetime", settingsSectionObject, maxTmpEntityLifetime)) {
        tree->setEntityMaxTmpLifetime(maxTmpEntityLifetime);
    } else {
        tree->setEntityMaxTmpLifetime(EntityTree::DEFAULT_MAX_TMP_ENTITY_LIFETIME);
    }

    tree->setWantEditLogging(wantEditLogging);
    tree->setWantTerseEditLogging(wantTerseEditLogging);

    QString entityScriptSourceWhitelist;
    if (readOptionString("entityScriptSourceWhitelist", settingsSectionObject, entityScriptSourceWhitelist)) {
        tree->setEntityScriptSourceWhitelist(entityScriptSourceWhitelist);
    } else {
        tree->setEntityScriptSourceWhitelist("");
    }

    if (readOptionString("entityEditFilter", settingsSectionObject, _entityEditFilter) && !_entityEditFilter.isEmpty()) {
        // Tell the tree that we have a filter, so that it doesn't accept edits until we have a filter function set up.
        std::static_pointer_cast<EntityTree>(_tree)->setHasEntityFilter(true);
        // Now fetch script from file asynchronously.
        QUrl scriptURL(_entityEditFilter);

        // The following should be abstracted out for use in Agent.cpp (and maybe later AvatarMixer.cpp)
        if (scriptURL.scheme().isEmpty() || (scriptURL.scheme() == URL_SCHEME_FILE)) {
            qWarning() << "Cannot load script from local filesystem, because assignment may be on a different computer.";
            scriptRequestFinished();
            return;
        }
        auto scriptRequest = ResourceManager::createResourceRequest(this, scriptURL);
        if (!scriptRequest) {
            qWarning() << "Could not create ResourceRequest for Agent script at" << scriptURL.toString();
            scriptRequestFinished();
            return;
        }
        // Agent.cpp sets up a timeout here, but that is unnecessary, as ResourceRequest has its own.
        connect(scriptRequest, &ResourceRequest::finished, this, &EntityServer::scriptRequestFinished);
        // FIXME: handle atp rquests setup here. See Agent::requestScript()
        qInfo() << "Requesting script at URL" << qPrintable(scriptRequest->getUrl().toString());
        scriptRequest->send();
        qDebug() << "script request sent";
    }
}

// Copied from ScriptEngine.cpp. We should make this a class method for reuse.
// Note: I've deliberately stopped short of using ScriptEngine instead of QScriptEngine, as that is out of project scope at this point.
static bool hasCorrectSyntax(const QScriptProgram& program) {
    const auto syntaxCheck = QScriptEngine::checkSyntax(program.sourceCode());
    if (syntaxCheck.state() != QScriptSyntaxCheckResult::Valid) {
        const auto error = syntaxCheck.errorMessage();
        const auto line = QString::number(syntaxCheck.errorLineNumber());
        const auto column = QString::number(syntaxCheck.errorColumnNumber());
        const auto message = QString("[SyntaxError] %1 in %2:%3(%4)").arg(error, program.fileName(), line, column);
        qCritical() << qPrintable(message);
        return false;
    }
    return true;
}
static bool hadUncaughtExceptions(QScriptEngine& engine, const QString& fileName) {
    if (engine.hasUncaughtException()) {
        const auto backtrace = engine.uncaughtExceptionBacktrace();
        const auto exception = engine.uncaughtException().toString();
        const auto line = QString::number(engine.uncaughtExceptionLineNumber());
        engine.clearExceptions();

        static const QString SCRIPT_EXCEPTION_FORMAT = "[UncaughtException] %1 in %2:%3";
        auto message = QString(SCRIPT_EXCEPTION_FORMAT).arg(exception, fileName, line);
        if (!backtrace.empty()) {
            static const auto lineSeparator = "\n    ";
            message += QString("\n[Backtrace]%1%2").arg(lineSeparator, backtrace.join(lineSeparator));
        }
        qCritical() << qPrintable(message);
        return true;
    }
    return false;
}
void EntityServer::scriptRequestFinished() {
    qDebug() << "script request completed";
    auto scriptRequest = qobject_cast<ResourceRequest*>(sender());
    const QString urlString = scriptRequest->getUrl().toString();
    if (scriptRequest && scriptRequest->getResult() == ResourceRequest::Success) {
        auto scriptContents = scriptRequest->getData();
        qInfo() << "Downloaded script:" << scriptContents;
        QScriptProgram program(scriptContents, urlString);
        if (hasCorrectSyntax(program)) {
            _entityEditFilterEngine.evaluate(scriptContents);
            if (!hadUncaughtExceptions(_entityEditFilterEngine, urlString)) {
                std::static_pointer_cast<EntityTree>(_tree)->initEntityEditFilterEngine(&_entityEditFilterEngine, [this]() {
                    return hadUncaughtExceptions(_entityEditFilterEngine, _entityEditFilter);
                });
                scriptRequest->deleteLater();
                qDebug() << "script request filter processed";
                return;
            }
        }
    } else if (scriptRequest) {
        qCritical() << "Failed to download script at" << urlString;
        // See HTTPResourceRequest::onRequestFinished for interpretation of codes. For example, a 404 is code 6 and 403 is 3. A timeout is 2. Go figure.
        qCritical() << "ResourceRequest error was" << scriptRequest->getResult();
    } else {
        qCritical() << "Failed to create script request.";
    }
    // Hard stop of the assignment client on failure. We don't want anyone to think they have a filter in place when they don't.
    // Alas, only indications will be the above logging with assignment client restarting repeatedly, and clients will not see any entities.
    qDebug() << "script request failure causing stop";
    stop();
}

void EntityServer::nodeAdded(SharedNodePointer node) {
    EntityTreePointer tree = std::static_pointer_cast<EntityTree>(_tree);
    tree->knowAvatarID(node->getUUID());
    OctreeServer::nodeAdded(node);
}

void EntityServer::nodeKilled(SharedNodePointer node) {
    EntityTreePointer tree = std::static_pointer_cast<EntityTree>(_tree);
    tree->deleteDescendantsOfAvatar(node->getUUID());
    tree->forgetAvatarID(node->getUUID());
    OctreeServer::nodeKilled(node);
}

// FIXME - this stats tracking is somewhat temporary to debug the Whiteboard issues. It's not a bad
// set of stats to have, but we'd probably want a different data structure if we keep it very long.
// Since this version uses a single shared QMap for all senders, there could be some lock contention 
// on this QWriteLocker
void EntityServer::trackSend(const QUuid& dataID, quint64 dataLastEdited, const QUuid& sessionID) {
    QWriteLocker locker(&_viewerSendingStatsLock);
    _viewerSendingStats[sessionID][dataID] = { usecTimestampNow(), dataLastEdited };
}

void EntityServer::trackViewerGone(const QUuid& sessionID) {
    QWriteLocker locker(&_viewerSendingStatsLock);
    _viewerSendingStats.remove(sessionID);
    if (_entitySimulation) {
        _entitySimulation->clearOwnership(sessionID);
    }
}

QString EntityServer::serverSubclassStats() {
    QLocale locale(QLocale::English);
    QString statsString;

    // display memory usage stats
    statsString += "<b>Entity Server Memory Statistics</b>\r\n";
    statsString += QString().sprintf("EntityTreeElement size... %ld bytes\r\n", sizeof(EntityTreeElement));
    statsString += QString().sprintf("       EntityItem size... %ld bytes\r\n", sizeof(EntityItem));
    statsString += "\r\n\r\n";

    statsString += "<b>Entity Server Sending to Viewer Statistics</b>\r\n";
    statsString += "----- Viewer Node ID -----------------    ----- Entity ID ----------------------    "
                   "---------- Last Sent To ----------    ---------- Last Edited -----------\r\n";

    int viewers = 0;
    const int COLUMN_WIDTH = 24;

    {
        QReadLocker locker(&_viewerSendingStatsLock);
        quint64 now = usecTimestampNow();

        for (auto viewerID : _viewerSendingStats.keys()) {
            statsString += viewerID.toString() + "\r\n";

            auto viewerData = _viewerSendingStats[viewerID];
            for (auto entityID : viewerData.keys()) {
                ViewerSendingStats stats = viewerData[entityID];

                quint64 elapsedSinceSent = now - stats.lastSent;
                double sentMsecsAgo = (double)(elapsedSinceSent / USECS_PER_MSEC);

                quint64 elapsedSinceEdit = now - stats.lastEdited;
                double editMsecsAgo = (double)(elapsedSinceEdit / USECS_PER_MSEC);

                statsString += "                                          "; // the viewerID spacing
                statsString += entityID.toString();
                statsString += "    ";
                statsString += QString("%1 msecs ago")
                    .arg(locale.toString((double)sentMsecsAgo).rightJustified(COLUMN_WIDTH, ' '));
                statsString += QString("%1 msecs ago")
                    .arg(locale.toString((double)editMsecsAgo).rightJustified(COLUMN_WIDTH, ' '));
                statsString += "\r\n";
            }
            viewers++;
        }
    }
    if (viewers < 1) {
        statsString += "    no viewers... \r\n";
    }
    statsString += "\r\n\r\n";

    return statsString;
}
