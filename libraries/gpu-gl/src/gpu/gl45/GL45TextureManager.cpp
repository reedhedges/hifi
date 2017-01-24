//
//  Created by Bradley Austin Davis on 2016/04/03
//  Copyright 2013-2016 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//
#include "GL45TextureManager.h"

#if 0
#include <gl/GLHelpers.h>
#include <gl/Context.h>

#include <gpu/GPULogging.h>

#include "../gl/GLShared.h"
#include "../gl/GLTexture.h"

using namespace gpu;
using namespace gpu::gl;

GLTextureTransferHelper::GLTextureTransferHelper() {
#ifdef THREADED_TEXTURE_TRANSFER
    setObjectName("TextureTransferThread");
    _context.create();
    initialize(true, QThread::LowPriority);
    // Clean shutdown on UNIX, otherwise _canvas is freed early
    connect(qApp, &QCoreApplication::aboutToQuit, [&] { terminate(); });
#else
    initialize(false, QThread::LowPriority);
#endif
}

GLTextureTransferHelper::~GLTextureTransferHelper() {
#ifdef THREADED_TEXTURE_TRANSFER
    if (isStillRunning()) {
        terminate();
    }
#else
    terminate();
#endif
}

void GLTextureTransferHelper::transferTexture(const gpu::TexturePointer& texturePointer) {
    GLTexture* object = Backend::getGPUObject<GLTexture>(*texturePointer);

    Backend::incrementTextureGPUTransferCount();
    object->setSyncState(GLSyncState::Pending);
    Lock lock(_mutex);
    _pendingTextures.push_back(texturePointer);
}

void GLTextureTransferHelper::setup() {
    _context.makeCurrent();
}

void GLTextureTransferHelper::shutdown() {
    _context.makeCurrent();
}

void GLTextureTransferHelper::queueExecution(VoidLambda lambda) {
    Lock lock(_mutex);
    _pendingCommands.push_back(lambda);
}

#define MAX_TRANSFERS_PER_PASS 2

bool GLTextureTransferHelper::process() {
    PROFILE_RANGE(render_gpu_gl, __FUNCTION__)
    // Take any new textures or commands off the queue
    VoidLambdaList pendingCommands;
    TextureList newTransferTextures;
    {
        Lock lock(_mutex);
        newTransferTextures.swap(_pendingTextures);
        pendingCommands.swap(_pendingCommands);
    }

    if (!pendingCommands.empty()) {
        for (auto command : pendingCommands) {
            command();
        }
        glFlush();
    }


    if (!newTransferTextures.empty()) {
        for (auto& texturePointer : newTransferTextures) {
            GLTexture* object = Backend::getGPUObject<GLTexture>(*texturePointer);
            object->startTransfer();
            _transferringTextures.push_back(texturePointer);
        }

        recomputeTextureLoad();
    }

    // No transfers in progress, sleep
    if (_transferringTextures.empty()) {
#ifdef THREADED_TEXTURE_TRANSFER
        QThread::usleep(1);
#endif
        return true;
    }

    PROFILE_COUNTER_IF_CHANGED(render_gpu_gl, "transferringTextures", size_t, _transferringTextures.size())

    size_t transferCount = 0;
    for (auto textureIterator = _transferringTextures.begin(); textureIterator != _transferringTextures.end();) {
        if (++transferCount > MAX_TRANSFERS_PER_PASS) {
            break;
        }
        auto texture = *textureIterator;
        GLTexture* gltexture = Backend::getGPUObject<GLTexture>(*texture);
        if (gltexture->continueTransfer()) {
            ++textureIterator;
            continue;
        }

        gltexture->finishTransfer();

#ifdef THREADED_TEXTURE_TRANSFER
        clientWait();
#endif

        gltexture->_contentStamp = gltexture->_gpuObject.getDataStamp();
        gltexture->updateSize();
        gltexture->setSyncState(gpu::gl::GLSyncState::Transferred);
        Backend::decrementTextureGPUTransferCount();
        _transferredTextures.push_back(texture);
        textureIterator = _transferringTextures.erase(textureIterator);
    }

#ifdef THREADED_TEXTURE_TRANSFER
    if (!_transferringTextures.empty()) {
        // Don't saturate the GPU
        clientWait();
    } else {
        // Don't saturate the CPU
        QThread::msleep(1);
    }
#endif

    return true;
}

void GLTextureTransferHelper::recomputeTextureLoad() {
    PROFILE_RANGE(render_gpu_gl, __FUNCTION__)
    size_t allTexturesTargetMemory = 0;

    struct TextureAndSize {
        gpu::TexturePointer texture;
        size_t size;
        bool operator<(const TextureAndSize& other) const {
            return size < other.size;
        }
    };

    std::priority_queue<TextureAndSize> allTextures;
    {
        PROFILE_RANGE(render_gpu_gl, "BuildPriorityQueue")
        for (auto itr = _transferredTextures.begin(); itr != _transferredTextures.end();) {
            auto texturePointer = itr->lock();
            // Clear any deleted textures
            if (!texturePointer) {
                itr = _transferredTextures.erase(itr);
                continue;
            }

            GLTexture* object = Backend::getGPUObject<GLTexture>(*texturePointer);
            auto targetMemory = object->getTargetGpuSize();
            allTexturesTargetMemory += targetMemory;
            if (object->derezable()) {
                allTextures.push({ texturePointer, targetMemory });
            }
            ++itr;
        }

        for (auto texturePointer : _transferringTextures) {
            GLTexture* object = Backend::getGPUObject<GLTexture>(*texturePointer);
            auto targetMemory = object->getTargetGpuSize();
            allTexturesTargetMemory += targetMemory;
            if (object->derezable()) {
                allTextures.push({ texturePointer, targetMemory });
            }
        }
    }

    std::set<TexturePointer> demotedLiveTextures;
    auto allowedTextureMemory = GLTexture::getAllowedTextureSize();
    while (!allTextures.empty() && allTexturesTargetMemory > allowedTextureMemory) {
        const auto derezTarget = allTextures.top();
        allTextures.pop();
        auto texture = derezTarget.texture;

        GLTexture* object = Backend::getGPUObject<GLTexture>(*texture);
        // Find out if the mip removal will target a live texture mip level
        // Find out how much memory will be freed;
        auto derezData = object->preDerez();
        size_t freedMemory = derezData.first;
        bool liveDerez = derezData.second;
        if (liveDerez) {
            demotedLiveTextures.insert(texture);
        } else {
            object->derez();
        }
        // if the texture has more mip levels to lose, put it back into the queue
        if (object->derezable()) {
            allTextures.push({ texture, derezTarget.size - freedMemory });
        }
        allTexturesTargetMemory -= freedMemory;
    }

    // FIXME for all of the demoted live textures, enqueue an operation on the main thread to sync 
    // the min mip level to the new target mip level and block on that operations completion with 
    // a fence

#if 0
    if (GLTexture::getMemoryPressure() < 1.0f) {
        return;
    }

    Lock lock(texturesByMipCountsMutex);
    if (texturesByMipCounts.empty()) {
        // No available textures to derez
        return;
    }

    auto mipLevel = texturesByMipCounts.rbegin()->first;
    if (mipLevel <= 1) {
        // No mips available to remove
        return;
    }

    GL45Texture* targetTexture = nullptr;
    {
        auto& textures = texturesByMipCounts[mipLevel];
        assert(!textures.empty());
        targetTexture = *textures.begin();
}
    lock.unlock();
    targetTexture->derez();
#endif

    _transferringTextures.sort([](const gpu::TexturePointer& a, const gpu::TexturePointer& b)->bool {
        return a->getSize() < b->getSize();
    });
}

#endif