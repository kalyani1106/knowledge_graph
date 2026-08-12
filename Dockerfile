FROM node:20-alpine

WORKDIR /src

COPY package*.json ./

RUN npm install --legacy-peer-deps
RUN npm install @babel/runtime --save --legacy-peer-deps

COPY . .

RUN npm run setup

EXPOSE 3000

CMD ["npm", "start"]