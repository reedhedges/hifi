//
//  Created by Bradley Austin Davis on 2016/04/03
//  Copyright 2013-2016 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//
#ifndef hifi_gpu_gl_GLTextureTransfer_h
#define hifi_gpu_gl_GLTextureTransfer_h

#include <QtGlobal>
#include <QtCore/QSharedPointer>

#include <GenericQueueThread.h>

#include <gl/Context.h>

#include "../gl/GLShared.h"

namespace gpu { namespace gl {

using TextureList = std::list<TexturePointer>;
using TextureListIterator = TextureList::iterator;
using TextureWeakPointer = std::weak_ptr<gpu::Texture>;
using TextureWeakList = std::list<TextureWeakPointer>;

class GLTextureTransferHelper {
public:
    using VoidLambda = std::function<void()>;
    using VoidLambdaList = std::list<VoidLambda>;
    using Pointer = std::shared_ptr<GLTextureTransferHelper>;
    GLTextureTransferHelper();
    ~GLTextureTransferHelper();
    void transferTexture(const gpu::TexturePointer& texturePointer);
    bool process();

private:
    void recomputeTextureLoad();
    // Textures that have been submitted for transfer
    TextureList _pendingTextures;
    // Textures currently in the transfer process
    // Only used on the transfer thread
    TextureList _transferringTextures;
    TextureWeakList _transferredTextures;
};

} } // namespace gpu::gl 

#endif